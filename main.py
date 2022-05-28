import time

import numpy as np
import pygame
import taichi as ti
import taichi_glsl as ts
from taichi_glsl import vec2, vec3, vec4

ti.init(arch=ti.opengl)
resolution = width, height = vec2(1280, 720)


@ti.func
def perc(whole, number):
    return number / (whole / 100)


@ti.data_oriented
class PyShader:
    def __init__(self, _app):
        self.app = _app
        self.screen_array = np.full((width, height, 3), [0, 0, 0], np.uint8)
        # taichi fields
        self.screen_field = ti.Vector.field(3, int, (width, height))

        # GLOBAL FRAGMENT SHADER CONSTANTS

        # Compute shader data
        self.cs_data = ti.Vector.field(1, float, (width, height))

    @ti.kernel
    def render(self, time: float):
        """Fragment shader imitation"""
        for frag_coord in ti.grouped(self.screen_field):
            value = (self.cs_data[frag_coord.x, frag_coord.y][0] + 1) / 2
            self.screen_field[frag_coord.x, resolution.y - frag_coord.y] = vec3(value * 255, value * 255, value * 255)

    def update(self, data):
        _time = pygame.time.get_ticks() * 0.001
        self.cs_data.from_numpy(data)
        self.render(_time)

        # TODO: Some how fix this bottleneck: extremely slow from potential 400 -> 40
        self.screen_array = self.screen_field.to_numpy()

    def draw(self):
        pygame.surfarray.blit_array(self.app.screen, self.screen_array)

    def run(self, data):
        self.update(data)
        self.draw()


@ti.func
def random(x: float, y: float):
    st = vec2(x, y)
    return ts.fract(ts.sin(ts.dot(st.xy, vec2(12.9898, 78.233))) * 43758.5453123)


"""
Simplex noise
"""


@ti.func
def mod289vec3(x: float, y: float, z: float):
    new_x = vec3(x, y, z)
    return new_x - ts.floor(new_x * (1.0 / 289.0)) * 289.0


@ti.func
def mod289vec2(x: float, y: float):
    new_x = vec2(x, y)
    return new_x - ts.floor(new_x * (1.0 / 289.0)) * 289.0


@ti.func
def permute(x: float, y: float, z: float):
    new_x = vec3(x, y, z)
    perm_b = ((new_x * 34.0) + 10.0) * new_x
    return mod289vec3(perm_b.x, perm_b.y, perm_b.z)


@ti.func
def snoise(param_x: float, param_y: float):
    v = vec2(param_x, param_y)
    C = vec4(0.211324865405187, 0.366025403784439, -0.577350269189626, 0.024390243902439)
    i = ts.floor(v + ts.dot(v, C.yy))
    x0 = v - i + ts.dot(i, C.xx)

    i1 = vec2(1.0, 0.0) if x0.x > x0.y else vec2(0.0, 1.0)

    x12 = x0.xyxy + C.xxzz
    x12.xy -= i1

    i = mod289vec2(i.x, i.y)
    perm1_prep = i.y + vec3(0.0, i1.y, 1.0)
    perm1 = permute(perm1_prep.x, perm1_prep.y, perm1_prep.z) + i.x + vec3(0.0, i1.x, 1.0)
    p = permute(perm1.x, perm1.y, perm1.z)
    m = max(0.5 - vec3(ts.dot(x0, x0), ts.dot(x12.xy, x12.xy), ts.dot(x12.zw, x12.zw)), 0.0)
    m = m * m
    m = m * m

    x = 2.0 * ts.fract(p * C.www) - 1.0
    h = abs(x) - 0.5
    ox = ts.floor(x + 0.5)
    a0 = x - ox

    m *= 1.79284291400159 - 0.85373472095314 * (a0 * a0 + h * h)

    g = vec3(0.0, 0.0, 0.0)
    g.x = a0.x * x0.x + h.x * x0.y
    g.yz = a0.yz * x12.xz + h.yz * x12.yw
    return 130.0 * ts.dot(m, g)


# @ti.data_oriented
# class ComputeShader:
#    def __init__(self, app, compute_area, seed, octaves, _scale=1):
#        self.app = app
#        self.compute_area = compute_area
#        self.compute_array = np.full(self.compute_area, 0.0, np.float64)
#
#        # taichi fields
#        self.screen_field = ti.Vector.field(1, float, self.compute_area)
#
#        # GLOBAL FRAGMENT SHADER CONSTANTS
#        self.seed = seed
#        self.OFFSETS = [0, 0]
#        self.octaves = octaves
#        self.scale = _scale
#
#    @ti.kernel
#    def render(self, time: float, x_offset: float, y_offset: float, seed: int, octaves: int, scale: float, const_cng: float):
#        """Fragment shader imitation"""
#        for frag_coord in ti.grouped(self.screen_field):
#            # random_seed = random(seed, seed)
#            #
#            # noise = 1 * snoise((p.x + random_seed + x_offset) * 1, (p.y + random_seed + y_offset) * 1) + \
#            #        0.5 * snoise((p.x + random_seed + x_offset) * 2, (p.y + random_seed + y_offset) * 2) + \
#            #        0.25 * snoise((p.x + random_seed + x_offset) * 4, (p.y + random_seed + y_offset) * 4) + \
#            #        0.125 * snoise((p.x + random_seed + x_offset) * 8, (p.y + random_seed + y_offset) * 8) + \
#            #        0.0625 * snoise((p.x + random_seed + x_offset) * 16, (p.y + random_seed + y_offset) * 16) + \
#            #        0.03125 * snoise((p.x + random_seed + x_offset) * 32, (p.y + random_seed + y_offset) * 32) + \
#            #        0.015625 * snoise((p.x + random_seed + x_offset) * 64, (p.y + random_seed + y_offset) * 64) + \
#            #        0.0078125 * snoise((p.x + random_seed + x_offset) * 128, (p.y + random_seed + y_offset) * 128)
#            #
#            # self.screen_field[frag_coord.x, frag_coord.y] = [
#            #    noise / (1 + 0.5 + 0.25 + 0.125 + 0.0625 + 0.03125 + 0.015625 + 0.0078125)]
#            p = (frag_coord.xy / resolution.y) * 2.0 - 1.0
#
#            random_seed = random(seed, seed)
#
#            last_mult = 1.0
#            last_mult_sum = 0.0
#            last_bmult = 1
#
#            noise = 0.0
#
#            for x in range(octaves):
#                if x != 0:
#                    last_mult = last_mult / 2
#                    last_mult_sum += last_mult
#                    last_bmult = last_bmult + last_bmult
#                else:
#                    last_mult_sum += 1
#                noise += last_mult * snoise(((p.x + random_seed + x_offset) * last_bmult) * scale,
#                                            ((p.y + random_seed + y_offset) * last_bmult) * scale)
#
#            self.screen_field[frag_coord.x, frag_coord.y] = [noise / last_mult_sum]
#
#    def update(self):
#        time = pygame.time.get_ticks() * 0.001
#        self.render(time, self.OFFSETS[0], self.OFFSETS[1], self.seed, self.octaves, self.scale, -0.65)
#
#    def swap(self):
#        self.compute_array = self.screen_field.to_numpy()
#
#    def run(self) -> np.full:
#        self.update()
#        self.swap()
#        return self.compute_array.copy()


@ti.data_oriented
class Noise:
    def __init__(self, app, octaves, persistence, lacunarity, _scale=1, seed=0):
        self.app = app
        self.compute_array = np.full(resolution, 0.0, np.float64)

        # taichi fields
        self.screen_field = ti.Vector.field(1, float, resolution)

        # GLOBAL FRAGMENT SHADER CONSTANTS
        self.seed = seed
        self.OFFSETS = [0, 0]

        self.persistence = persistence
        self.lacunarity = lacunarity
        self.octaves = octaves
        self.scale = _scale

    @ti.kernel
    def render(self, time: float, x_offset: float, y_offset: float, seed: int, octaves: int, per: float, lac: float):
        """Fragment shader imitation"""
        for frag_coord in ti.grouped(self.screen_field):
            p = (frag_coord.xy / resolution.y) * 2.0 - 1.0
            freq = 1.0
            amp = 1.0
            _max = 1.0
            x = p.x + seed + x_offset
            y = p.y + seed + y_offset
            total = snoise(x, y)
            for i in range(octaves):
                freq *= lac
                amp *= per
                _max += amp
                total += snoise(x * freq, y * freq) * amp
            self.screen_field[frag_coord.x, frag_coord.y] = [total / _max]

    def update(self):
        time = pygame.time.get_ticks() * 0.001
        self.render(time, self.OFFSETS[0], self.OFFSETS[1], self.seed, self.octaves, self.persistence, self.lacunarity)

    def swap(self):
        self.compute_array = self.screen_field.to_numpy()

    def run(self) -> np.full:
        self.update()
        self.swap()
        return self.compute_array.copy()


position = vec2(0, 0)


class App:
    def __init__(self):
        self.screen = pygame.display.set_mode(resolution)
        self.clock = pygame.time.Clock()
        self.compute_shader = Noise(self, 8, 0.5, 2.0)
        self.shader = PyShader(self)
        self.SPEED = 0.05

    def run(self):
        while True:
            comp = self.compute_shader.run()
            self.shader.run(comp)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit(0)

            keys = pygame.key.get_pressed()
            if keys[pygame.K_w]:
                position.y += self.SPEED
            if keys[pygame.K_s]:
                position.y -= self.SPEED
            if keys[pygame.K_a]:
                position.x -= self.SPEED
            if keys[pygame.K_d]:
                position.x += self.SPEED

            self.compute_shader.OFFSETS = position

            self.clock.tick(0)
            pygame.display.flip()
            pygame.display.set_caption(f"FPS: {self.clock.get_fps() :.2f}")


if __name__ == '__main__':
    app = App()
    app.run()
