from bs4 import BeautifulSoup
import gymnasium as gym
import numpy as np


def make_env(env_name: str,
        config={},
        #mujoco_config={},
        meta_config=None) -> gym.Env:
    env = gym.make(env_name, render_mode='rgb_array', **config)
    if meta_config is not None:
        #env = MetaWrapper(env, **meta_config)
        raise NotImplementedError()
    return env


DEFAULT_CAMERA_CONFIG = {
    "distance": 4.0,
}


class MjcfModelFactory():
    def __init__(self, model_name):
        self.grid_size = (10, 10)
        self.cell_size = 2
        self.height = 3

        self.soup = BeautifulSoup(features='xml')

        self.root = self._create_root(model_name)
        self.default = self._create_default()
        self.asset = self._create_asset()
        self.worldbody = self.soup.new_tag('worldbody')
        self.actuator = self.soup.new_tag('actuator')

        self.root.append(self.default)
        self.root.append(self.asset)
        self.root.append(self.worldbody)
        self.root.append(self.actuator)

        self.agent = None # Body that can be controlled
        self.walls = []
        self.balls = []
        self.boxes = []

    def to_xml(self):
        return self.soup.prettify()

    def _create_root(self, model_name):
        root = self.soup.new_tag('mujoco', attrs={'model': model_name})

        root.append(self.soup.new_tag('compiler', attrs={
            'angle': 'degree',
            'coordinate': 'local',
            'inertiafromgeom': 'true',
        }))
        root.append(self.soup.new_tag('option', attrs={
            'integrator': 'RK4',
            'timestep': '0.01'
        }))

        self.soup.append(root)

        return root

    def _create_default(self):
        default = self.soup.new_tag('default')
        default.append(self.soup.new_tag('joint', attrs={
            'armature': '1',
            'damping': '1',
            'limited': 'true'
        }))
        default.append(self.soup.new_tag('geom', attrs={
            'conaffinity': '0',
            'condim': '3',
            'density': '5',
            'friction': '1 0.5 0.5',
            'margin': '0.01',
            'rgba': '0.8 0.6 0.4 1',
        }))

        return default

    def _create_asset(self):
        asset = self.soup.new_tag('asset')
        asset.append(self.soup.new_tag('texture', attrs={
            'builtin': 'gradient',
            'height': '100',
            'rgb1': '1 1 1',
            'rgb2': '0 0 0',
            'type': 'skybox',
            'width': '100'
        }))
        asset.append(self.soup.new_tag('texture', attrs={
            'builtin': 'flat',
            'height': '1278',
            'mark': 'cross',
            'markrgb': '1 1 1',
            'name': 'texgeom',
            'random': '0.01',
            'rgb1': '0.8 0.6 0.4',
            'rgb2': '0.8 0.6 0.4',
            'type': 'cube',
            'width': '127'
        }))
        asset.append(self.soup.new_tag('texture', attrs={
            'builtin': 'checker',
            'height': '100',
            'name': 'texplane',
            'rgb1': '0 0 0',
            'rgb2': '0.8 0.8 0.8',
            'type': '2d',
            'width': '100'
        }))
        asset.append(self.soup.new_tag('material', attrs={
            'name': 'MatPlane',
            'reflectance': '0.5',
            'shininess': '1',
            'specular': '1',
            'texrepeat': '60 60',
            'texture': 'texplane'
        }))
        asset.append(self.soup.new_tag('material', attrs={
            'name': 'geom',
            'texture': 'texgeom',
            'texuniform': 'true'
        }))

        return asset

    def add_ground(self):
        self.worldbody.append(self.soup.new_tag('geom', attrs={
            'conaffinity': '1',
            'condim': '3',
            'material': 'MatPlane',
            'name': 'floor',
            'pos': '0 0 0',
            'rgba': '0.8 0.9 0.8 1',
            'size': '40 40 40',
            'type': 'plane',
        }))

    def add_ceiling(self):
        self.worldbody.append(self.soup.new_tag('geom', attrs={
            'conaffinity': '1',
            'condim': '3',
            'material': 'MatPlane',
            'name': 'floor',
            'pos': f'0 0 {self.height}',
            'rgba': '0 0 0 0',
            'size': '40 40 40',
            'type': 'plane',
        }))

    def add_light(self):
        self.worldbody.append(self.soup.new_tag('light', attrs={
            'cutoff': '100',
            'diffuse': '1 1 1',
            'dir': '0 0 -1',
            'exponent': '1',
            'pos': '0 0 3',
            'specular': '1 1 1'
        }))

    def add_ant(self, pos, name_prefix='', show_nose = False, num_legs=4):
        body_name = f'{name_prefix}torso'
        body = self.soup.new_tag('body', attrs={
            'name': body_name,
            'pos': f'{pos[0] * self.cell_size} {pos[1] * self.cell_size} 0.75'
        })
        actuators = []

        camera_track = self.soup.new_tag(
            'camera',
            attrs={
                'name': name_prefix+'track',
                'mode': 'trackcom',
                'pos': '0 -3 0.3',
                'xyaxes': '1 0 0 0 0 1'
            }
        )
        body.append(camera_track)

        camera_first_person = self.soup.new_tag('camera', attrs={'name': name_prefix+'first_person', 'mode': 'fixed', 'pos': '0 0 0', 'axisangle': '1 0 0 90'})
        body.append(camera_first_person)

        torso_geom = self.soup.new_tag('geom', attrs={'name': name_prefix+'torso_geom', 'pos': '0 0 0', 'size': '0.25', 'type': 'sphere'})
        body.append(torso_geom)

        # Make it a free joint so that we can move it around
        root_joint = self.soup.new_tag('joint', attrs={'armature': '0', 'damping': '0', 'limited': 'false', 'margin': '0.01', 'name': name_prefix+'root_joint', 'pos': '0 0 0', 'type': 'free'})
        body.append(root_joint)

        # Nose (for debugging purposes. Shows the direction of the first person camera.)
        if show_nose:
            body.append(self.soup.new_tag('geom', attrs={
                'contype': '0',
                'conaffinity': '0',
                'name': name_prefix+'nose_geom',
                'fromto': '0 0 0.1 0 10 0',
                'size': '0.01',
                'type': 'capsule',
                'rgba': '1 0 0 0.5',
                'mass': '0',
            }))

        # Legs
        angles = np.linspace(0, 360, num_legs, endpoint=False)
        for i in range(num_legs):
            leg = self.soup.new_tag('body', attrs={
                'name': f'{name_prefix}leg{i}',
                'pos': '0 0 0',
                'axisangle': f'0 0 1 {angles[i]}'
            })
            body.append(leg)

            leg_aux_geom = self.soup.new_tag('geom', attrs={
                'fromto': '0.0 0.0 0.0 0.2 0.2 0.0',
                'name': f'{name_prefix}leg{i}_aux_geom',
                'size': '0.08',
                'type': 'capsule'
            })
            leg.append(leg_aux_geom)

            leg_aux = self.soup.new_tag('body', attrs={
                'name': f'{name_prefix}leg{i}_aux1',
                'pos': '0.2 0.2 0'
            })
            leg.append(leg_aux)

            leg_hip = self.soup.new_tag('joint', attrs={
                'axis': f'0 0 1',
                'name': f'{name_prefix}hip{i}',
                'pos': '0.0 0.0 0.0',
                'range': '-30 30',
                'type': 'hinge'
            })
            leg_aux.append(leg_hip)
            actuators.append(self.soup.new_tag('motor', attrs={
                'ctrllimited': 'true',
                'ctrlrange': '-1 1',
                'joint': f'{name_prefix}hip{i}',
                'gear': '150',
            }))

            leg_geom = self.soup.new_tag('geom', attrs={
                'fromto': '0.0 0.0 0.0 0.2 0.2 0.0',
                'name': f'{name_prefix}leg{i}_geom',
                'size': '0.08',
                'type': 'capsule'
            })
            leg_aux.append(leg_geom)

            leg_body = self.soup.new_tag('body', attrs={'pos': '0.2 0.2 0', 'name': f'{name_prefix}leg{i}_aux2'})
            leg_aux.append(leg_body)

            leg_ankle = self.soup.new_tag('joint', attrs={
                'axis': '-1 1 0',
                'name': f'{name_prefix}ankle{i}',
                'pos': '0.0 0.0 0.0',
                'range': '30 70',
                'type': 'hinge'
            })
            leg_body.append(leg_ankle)
            actuators.append(self.soup.new_tag('motor', attrs={
                'ctrllimited': 'true',
                'ctrlrange': '-1 1',
                'joint': f'{name_prefix}ankle{i}',
                'gear': '150',
            }))

            leg_ankle_geom = self.soup.new_tag('geom', attrs={
                'fromto': '0.0 0.0 0.0 0.4 0.4 0.0',
                'name': f'{name_prefix}ankle{i}_geom',
                'size': '0.08',
                'type': 'capsule'
            })
            leg_body.append(leg_ankle_geom)

        self.worldbody.append(body)
        for a in actuators:
            self.actuator.append(a)
        self.agent = { 'body': body, 'actuators': actuators }

        return body_name

    def add_wall(self, cell, cell2=None):
        if cell2 is None:
            pos = (
                cell[0] * self.cell_size,
                cell[1] * self.cell_size,
                self.height / 2,
            )
            size = (self.cell_size / 2, self.cell_size / 2, self.height / 2)
        else:
            pos = (
                (cell[0] + cell2[0]) * self.cell_size / 2,
                (cell[1] + cell2[1]) * self.cell_size / 2,
                self.height / 2,
            )
            size = (
                (abs(cell[0] - cell2[0]) + 1) * self.cell_size / 2,
                (abs(cell[1] - cell2[1]) + 1) * self.cell_size / 2,
                self.height / 2,
            )
        #print(cell, cell2, pos, size)
        #self.worldbody.append(self.soup.new_tag('geom', attrs={
        #    'conaffinity': '0',
        #    'condim': '3',
        #    'size': '0.08',
        #    'fromto': f'{pos[0]} {pos[1]} 0 {pos[0]} {pos[1]} 5',
        #    'type': 'capsule',
        #    'rgba': '0 1 0 1',
        #}))
        wall = self.soup.new_tag('geom', attrs={
            'conaffinity': '1',
            'condim': '3',
            'material': 'MatPlane',
            'name': f'wall_{len(self.walls)}',
            'pos': ' '.join([str(p) for p in pos]),
            'size': ' '.join([str(s) for s in size]),
            'type': 'box',
            'rgba': '0.8 0.8 0.8 0.5',
        })
        self.walls.append(wall)
        self.worldbody.append(wall)

    def add_ball(self, pos, size=0.5, rgba=[1, 0, 0, 1]):
        idx = len(self.balls)
        size = 0.5
        body_name = f'ball{idx}'

        coords = [pos[0] * self.cell_size, pos[1] * self.cell_size, size]
        body = self.soup.new_tag('body', attrs={'name': body_name, 'pos': ' '.join(map(str, coords))})

        body.append(self.soup.new_tag('joint', attrs={
            'armature': '0',
            'damping': '0',
            'limited': 'false',
            'margin': '0.01',
            'name': f'ball{idx}_joint',
            'pos': '0 0 0',
            'type': 'free'
        }))
        body.append(self.soup.new_tag('geom', attrs={
            'conaffinity': '1',
            'name': f'ball{idx}_geom',
            'pos': '0 0 0',
            'size': str(size),
            'type': 'sphere',
            'rgba': ' '.join(map(str, rgba))
        }))

        self.balls.append(body)
        self.worldbody.append(body)

        return body_name

    def add_box(self, pos, size=0.5, rgba=[1, 0, 0, 1]):
        idx = len(self.boxes)
        size = 0.5
        body_name = f'box{idx}'

        coords = [pos[0] * self.cell_size, pos[1] * self.cell_size, size]
        body = self.soup.new_tag('body', attrs={'name': body_name, 'pos': ' '.join(map(str, coords))})

        body.append(self.soup.new_tag('joint', attrs={
            'armature': '0',
            'damping': '0',
            'limited': 'false',
            'margin': '0.01',
            'name': f'box{idx}_joint',
            'pos': '0 0 0',
            'type': 'free'
        }))
        body.append(self.soup.new_tag('geom', attrs={
            'conaffinity': '1',
            'name': f'box{idx}_geom',
            'pos': '0 0 0',
            'size': f'{size} {size} {size}',
            'type': 'box',
            'rgba': ' '.join(map(str, rgba))
        }))

        self.balls.append(body)
        self.worldbody.append(body)

        return body_name


def strz(x):
    """ Truncate a null-terminated string. """
    for i,c in enumerate(x):
        if c == 0:
            return x[:i]
    return x


def list_geoms_by_body(env, body_id=0, depth=0, covered_bodies=None):
    if covered_bodies is None:
        covered_bodies = set()
    if body_id in covered_bodies:
        return
    covered_bodies.add(body_id)

    body_name_adr = env.model.name_bodyadr[body_id]
    body_name = strz(env.model.names[body_name_adr:])
    print(f'{"  "*depth} {body_id} {body_name}')
    for geom_id,geom_name_adr in enumerate(env.model.name_geomadr):
        geom_name = strz(env.model.names[geom_name_adr:])
        if env.model.geom_bodyid[geom_id] == body_id:
            print(f'{"  "*depth}   {geom_id} {geom_name}')
    for child_id,parent_id in enumerate(env.model.body_parentid):
        if parent_id == body_id:
            list_geoms_by_body(child_id, depth+1, covered_bodies)


def list_joints_with_ancestor(env, body_id, depth=0, covered_bodies=None):
    joint_ids = set()

    if covered_bodies is None:
        covered_bodies = set()
    if body_id in covered_bodies:
        return joint_ids
    covered_bodies.add(body_id)

    body_name_adr = env.model.name_bodyadr[body_id]
    body_name = strz(env.model.names[body_name_adr:])
    print(f'{"  "*depth} {body_id} {body_name}')
    for joint_id,joint_name_adr in enumerate(env.model.name_jntadr):
        joint_name = strz(env.model.names[joint_name_adr:])
        if env.model.jnt_bodyid[joint_id] == body_id:
            print(f'{"  "*depth}   {joint_id} {joint_name}')
            joint_ids.add(joint_id)
    for child_id,parent_id in enumerate(env.model.body_parentid):
        if parent_id == body_id:
            j = list_joints_with_ancestor(env, child_id, depth+1, covered_bodies)
            joint_ids.update(j)
    return joint_ids


def list_bodies_with_ancestor(env, body_id, depth=0):
    body_ids = set([body_id])

    body_name_adr = env.model.name_bodyadr[body_id]
    body_name = strz(env.model.names[body_name_adr:])
    print(f'{"  "*depth} {body_id} {body_name}')
    for child_id,parent_id in enumerate(env.model.body_parentid):
        if parent_id == body_id:
            body_ids.update(list_bodies_with_ancestor(env, child_id, depth+1))
    return body_ids


if __name__ == '__main__':
    #env = gym.make('AntFetchEnv-v0', render_mode='human')
    env = gym.make('AntBaselineEnv-v0', render_mode='human')
    env.reset()
    print(env.action_space)
    total_reward = 0
    for i in range(1000):
        env.render()
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())

        total_reward += reward
        done = terminated or truncated
        if done:
            print(f'Episode terminated with reward {total_reward}')
            total_reward = 0
            obs, info = env.reset()

        #collided_objects = env.get_collisions()
        #if len(collided_objects) > 0:
        #    print('Collisions:', collided_objects)
        #for obj in collided_objects:
        #    env.place_object(obj)


        ## Display image
        #import matplotlib
        #from matplotlib import pyplot as plt
        #plt.imshow(obs['image'].transpose(1,2,0))
        #plt.show()

        ## Save image
        #import imageio
        #imageio.imwrite(f'frame_{i:04d}.png', rgb_array)

        #breakpoint()
