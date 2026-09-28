"""Deterministic fixed-peg task; run with Isaac Sim's python.bat."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
CONFIG = {
    'physics_dt': 1/60, 'control_substeps': 4, 'max_steps': 180,
    'initial_joints_deg': [-90, -60, -100, -110, 90, 0],
    'hole_origin_m': [0.194, 0.371, 0.57], 'peg_tip_offset_m': 0.12,
    'peg_length_m': 0.08, 'peg_half_width_m': 0.004,
    'hole_half_width_m': 0.005, 'hole_length_m': 0.05,
    'position_randomization_m': [0.008, 0.008, 0.004],
    'rotation_randomization_rad': [0.015, 0.015, 0.04],
    'translation_limit_m': 0.003, 'rotation_limit_rad': 0.025,
    'min_depth_m': 0.025, 'max_depth_m': 0.035,
    'lateral_tolerance_m': 0.0007, 'angular_tolerance_rad': 0.01,
    'contact_force_limit_n': 1.0,
}


def terminal(state, contact, steps):
    state = np.asarray(state)
    if state.shape != (6,) or not np.isfinite(state).all():
        return 'NONFINITE_STATE'
    if not np.isfinite(contact) or contact > CONFIG['contact_force_limit_n']:
        return 'COLLISION'
    if (CONFIG['min_depth_m'] <= state[2] <= CONFIG['max_depth_m']
            and np.linalg.norm(state[:2]) <= CONFIG['lateral_tolerance_m']
            and np.linalg.norm(state[3:]) <= CONFIG['angular_tolerance_rad']):
        return 'SUCCESS'
    return 'TIMEOUT' if steps >= CONFIG['max_steps'] else None


def expert(state):
    """Approach clear of the lip; align all axes before entering the narrow hole."""
    target = np.zeros(6)
    target[2] = -0.035
    phase = 'APPROACH_ALIGN'
    if np.linalg.norm(state[:2]) < 0.0004 and np.linalg.norm(state[3:]) < 0.004:
        target[2] = 0.03
        phase = 'INSERT'
    delta = target - state
    action = np.r_[np.clip(delta[:3], -CONFIG['translation_limit_m'], CONFIG['translation_limit_m']),
                   np.clip(delta[3:], -CONFIG['rotation_limit_rad'], CONFIG['rotation_limit_rad'])]
    return action, phase


class InsertionEnv:
    def __init__(self, app):
        # SimulationApp must start before loading Kit modules.
        import omni.usd
        import omni.replicator.core as rep
        from pxr import Gf, UsdGeom, UsdLux, UsdPhysics, PhysxSchema
        from isaacsim.core.experimental.prims import Articulation, RigidPrim
        from isaacsim.core.simulation_manager import SimulationManager
        from isaacsim.storage.native import get_assets_root_path
        self.app, self.usd, self.rep = app, omni.usd, rep
        self.Gf, self.UG, self.UL, self.UP, self.PX = Gf, UsdGeom, UsdLux, UsdPhysics, PhysxSchema
        self.Articulation, self.RigidPrim, self.SM = Articulation, RigidPrim, SimulationManager
        self.asset = get_assets_root_path()+'/Isaac/Robots/UniversalRobots/ur10e/ur10e.usd'
        self.rgb = self.product = None

    def cube(self, path, position, scale, color):
        cube = self.UG.Cube.Define(self.stage, path)
        cube.CreateSizeAttr(1.0)
        cube.CreateDisplayColorAttr([self.Gf.Vec3f(*color)])
        xf = self.UG.XformCommonAPI(cube)
        xf.SetTranslate(self.Gf.Vec3d(*[float(x) for x in position]))
        xf.SetScale(self.Gf.Vec3f(*scale))
        self.UP.CollisionAPI.Apply(cube.GetPrim())
        # Tight geometric clearance; avoid the default centimeter contact margin.
        collision = self.PX.PhysxCollisionAPI.Apply(cube.GetPrim())
        collision.CreateContactOffsetAttr(0.0002)
        collision.CreateRestOffsetAttr(0.0)

    def reset(self, seed):
        import omni.timeline
        from isaacsim.core.experimental.utils.stage import is_stage_loading
        omni.timeline.get_timeline_interface().stop()
        if self.rgb is not None:
            self.rgb.detach()
            self.product.destroy()
        self.usd.get_context().new_stage()
        self.stage = self.usd.get_context().get_stage()
        self.UG.SetStageMetersPerUnit(self.stage, 1.0)
        self.UG.SetStageUpAxis(self.stage, self.UG.Tokens.z)
        self.SM.setup_simulation(dt=CONFIG['physics_dt'], device='cuda')
        robot = self.stage.DefinePrim('/World/Robot', 'Xform')
        robot.GetReferences().AddReference(self.asset)
        for i in range(300):
            self.app.update()
            if i > 25 and not is_stage_loading():
                break
        else:
            raise RuntimeError('ASSET_LOAD_TIMEOUT')
        rng = np.random.default_rng(seed)
        self.hole_p = np.array(CONFIG['hole_origin_m']) + rng.uniform(-1, 1, 3)*CONFIG['position_randomization_m']
        self.hole_r = Rotation.from_rotvec(rng.uniform(-1, 1, 3)*CONFIG['rotation_randomization_rad'])*Rotation.from_euler('x', np.pi)
        fixture = self.stage.DefinePrim('/World/Fixture', 'Xform')
        xf = self.UG.Xformable(fixture)
        xf.AddTranslateOp().Set(self.Gf.Vec3d(*self.hole_p))
        q = self.hole_r.as_quat(scalar_first=True)
        xf.AddOrientOp().Set(self.Gf.Quatf(float(q[0]), self.Gf.Vec3f(*q[1:])))
        self.UP.RigidBodyAPI.Apply(fixture).CreateKinematicEnabledAttr(True)
        h, t, length = CONFIG['hole_half_width_m'], 0.035, CONFIG['hole_length_m']
        for axis in range(2):
            for sign in [-1, 1]:
                pos = [0., 0., length/2]
                pos[axis] = sign*(h+t/2)
                scale = [2*h, 2*h, length]
                scale[axis] = t
                scale[1-axis] = 2*(h+t)
                self.cube(f'/World/Fixture/Wall{axis}_{"p" if sign>0 else "m"}',pos,scale,(0.22,0.32,0.4))
        self.cube('/World/Fixture/Bottom',[0,0,length+0.005],[0.08,0.08,0.01],(0.22,0.32,0.4))
        self.cube('/World/Floor',[0,0,-0.025],[3,3,0.05],(0.6,0.62,0.65))
        offset, peg_length = CONFIG['peg_tip_offset_m'], CONFIG['peg_length_m']
        self.cube('/World/Robot/wrist_3_link/Peg',[0,0,offset-peg_length/2],
                  [2*CONFIG['peg_half_width_m']]*2+[peg_length],(0.85,0.47,0.12))
        self.cube('/World/Robot/wrist_3_link/Holder',[0,0,0.023],[0.024,0.024,0.046],(0.28,0.3,0.32))
        pedestal_height=float(self.hole_p[2]-0.055)
        self.cube('/World/Pedestal',[self.hole_p[0],self.hole_p[1],pedestal_height/2],
                  [0.13,0.13,pedestal_height],(0.28,0.3,0.32))
        self.UL.DomeLight.Define(self.stage,'/World/Light').CreateIntensityAttr(1000.)
        links = [p.GetPath().pathString for p in self.stage.Traverse()
                 if p.HasAPI(self.UP.RigidBodyAPI) and str(p.GetPath()).startswith('/World/Robot/')
                 and p.GetName() != 'base_link']
        self.contacts = self.RigidPrim(links, contact_filter_paths=['/World/Fixture','/World/Floor','/World/Pedestal'])
        self.contacts.set_enabled_contact_tracking([True], threshold=0.)
        self.robot = self.Articulation('/World/Robot')
        self.robot.set_enabled_self_collisions([True])
        self.wrist = self.RigidPrim('/World/Robot/wrist_3_link')
        self.rep.orchestrator.set_capture_on_play(False)
        camera = self.rep.functional.create.camera(position=(1.5,1.7,1.45),look_at=(0.05,0.20,0.55),focal_length=35.)
        self.product = self.rep.create.render_product(camera,(640,480))
        self.rgb = self.rep.annotators.get('rgb')
        self.rgb.attach(self.product)
        omni.timeline.get_timeline_interface().play()
        for _ in range(20): self.app.update()
        self.q0 = np.deg2rad(CONFIG['initial_joints_deg']).astype(np.float32)
        self.robot.set_dof_gains(stiffnesses=[60000.]*6,dampings=[3000.]*6)
        self.robot.set_dof_positions(self.q0)
        self.robot.set_dof_velocities(np.zeros(6,dtype=np.float32))
        self.robot.set_dof_position_targets(self.q0)
        for _ in range(60): self.SM.step(update_fabric=True)
        self.steps, self.max_contact = 0, 0.
        self.jindex = self.robot.link_names.index('wrist_3_link')-1
        self.initial = self.state().copy()
        return self.initial

    def pose(self):
        p, q = self.wrist.get_world_poses()
        r = Rotation.from_quat(q.numpy()[0], scalar_first=True)
        return p.numpy()[0]+r.apply([0,0,CONFIG['peg_tip_offset_m']]), r

    def state(self):
        p, r = self.pose()
        return np.r_[self.hole_r.inv().apply(p-self.hole_p), (self.hole_r.inv()*r).as_rotvec()]

    def step(self, action):
        action = np.asarray(action, dtype=float)
        bounds = np.array([CONFIG['translation_limit_m']]*3+[CONFIG['rotation_limit_rad']]*3)
        if action.shape != (6,) or not np.isfinite(action).all() or np.any(np.abs(action)>bounds+1e-8):
            raise ValueError('INVALID_ACTION')
        p, r = self.pose()
        desired_p = p+self.hole_r.apply(action[:3])
        desired_r = self.hole_r*Rotation.from_rotvec(action[3:])*self.hole_r.inv()*r
        for _ in range(CONFIG['control_substeps']):
            p, r = self.pose()
            error = np.r_[desired_p-p,(desired_r*r.inv()).as_rotvec()]
            j = self.robot.get_jacobian_matrices().numpy()[0,self.jindex].astype(float)
            offset = r.apply([0,0,CONFIG['peg_tip_offset_m']])
            j[:3] += np.cross(j[3:].T,offset).T
            dq = j.T @ np.linalg.solve(j@j.T+1e-5*np.eye(6),error)
            q = self.robot.get_dof_positions().numpy()[0]
            target = q+np.clip(dq,-0.02,0.02)
            lo, hi = self.robot.get_dof_limits()
            self.robot.set_dof_position_targets(np.clip(target,lo.numpy()[0],hi.numpy()[0]).astype(np.float32))
            self.SM.step(update_fabric=True)
            force = self.contacts.get_contact_force_matrix(dt=CONFIG['physics_dt']).numpy()
            net_force = self.contacts.get_net_contact_forces(dt=CONFIG['physics_dt']).numpy()
            self.max_contact = max(self.max_contact,float(np.linalg.norm(force,axis=-1).max()),
                                   float(np.linalg.norm(net_force,axis=-1).max()))
        self.steps += 1
        state = self.state()
        return state, terminal(state,self.max_contact,self.steps)

    def image(self):
        for _ in range(10):
            self.rep.orchestrator.step(rt_subframes=2,delta_time=0.,pause_timeline=False)
            image = self.rgb.get_data()
            if getattr(image,'size',0):
                return np.asarray(image)[...,:3].copy()
        raise RuntimeError('RGB_CAPTURE_EMPTY')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds',type=int,nargs='+',default=[0,0,1])
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--collision-check',action='store_true',help='Intentionally miss the hole by 15 mm; expect physical contact failure.')
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    from isaacsim import SimulationApp
    app=SimulationApp({'headless':True,'width':640,'height':480})
    report={'status':'FAILED','config':CONFIG,'episodes':[],
            'test':'collision_rejection' if args.collision_check else 'expert_smoke'}
    try:
        from PIL import Image
        env=InsertionEnv(app)
        report['asset']=env.asset
        report['project_commit']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        report['project_dirty']=bool(subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip())
        import importlib.metadata
        report['environment']={name:importlib.metadata.version(name) for name in ('numpy','scipy','Pillow')}
        report['isaac_sim_version']=(Path(sys.executable).parents[2]/'VERSION').read_text().strip()
        report['script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        report['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total','--format=csv,noheader'],text=True).strip()
        for i,seed in enumerate(args.seeds):
            state=env.reset(seed)
            initial=state.tolist()
            Image.fromarray(env.image()).save(args.output/f'{i:02d}_start.png')
            outcome=None
            trace=[]
            while outcome is None:
                policy_state=state.copy()
                if args.collision_check:
                    policy_state[0]-=0.015
                action,phase=expert(policy_state)
                trace.append({'state':state.tolist(),'action':action.tolist(),'phase':phase})
                state,outcome=env.step(action)
            Image.fromarray(env.image()).save(args.output/f'{i:02d}_end.png')
            row={'seed':seed,'outcome':outcome,'steps':env.steps,'initial_state':initial,
                 'final_state':state.tolist(),'max_contact_n':env.max_contact}
            report['episodes'].append(row)
            (args.output/f'{i:02d}_trace.json').write_text(json.dumps(trace))
            print('EPISODE',json.dumps(row),flush=True)
        expected='COLLISION' if args.collision_check else 'SUCCESS'
        for first in range(len(report['episodes'])):
            for second in range(first):
                a,b=report['episodes'][first],report['episodes'][second]
                if a['seed']==b['seed'] and a!=b:
                    raise RuntimeError('RESET_NOT_DETERMINISTIC')
        report['status']='PASS' if all(x['outcome']==expected for x in report['episodes']) else 'FAILED'
    except Exception as exc:
        report['error']=str(exc)
        raise
    finally:
        (args.output/'result.json').write_text(json.dumps(report,indent=2,allow_nan=False))
        app.close()
    if report['status']!='PASS':
        sys.exit(1)


if __name__=='__main__':
    main()
