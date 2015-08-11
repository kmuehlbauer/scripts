#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import numpy as np
from vispy import app, gloo
from vispy.io import _data_dir

vertex = """
attribute vec2 a_position;
attribute vec2 a_texcoord;
varying vec2 v_texcoord;
void main (void)
{
    v_texcoord = a_texcoord;
    gl_Position = vec4(a_position,0.0,1.0);
}
"""

fragment = """
#include "misc/spatial-filters.frag"
uniform sampler2D u_texture;
uniform vec2 u_shape;
varying vec2 v_texcoord;
void main()
{
    const float level = 10.0;
    vec4 texel1 = Bicubic(u_texture, u_shape, v_texcoord);

    float si1 = abs(ceil(texel1.r * level)/level);
    vec4 col1 = vec4(si1,si1,si1,1.0);
    gl_FragColor = col1;
}
"""

data = np.zeros(4, dtype=[('a_position', np.float32, 2),
                          ('a_texcoord', np.float32, 2)])
data['a_position'] = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]])
data['a_texcoord'] = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
np.random.seed(1000)
I = np.random.uniform(0,1,(5,5)).astype(np.float32)

class Canvas(app.Canvas):
    def __init__(self):
        app.Canvas.__init__(self, keys='interactive', size=(512, 512))
        self.program = gloo.Program(vertex, fragment)
        self.program['u_texture'] = gloo.Texture2D(I)

        self.program.bind(gloo.VertexBuffer(data))

        self.program['u_kernel'] = np.load(os.path.join(_data_dir, 'spatial-filters.npy'))
        self.program['u_kernel'].interpolation = 'nearest'
        self.program['u_texture'].interpolation = 'nearest'
        self.program['u_texture'].wrapping = 'clamp_to_edge'
        self.program['u_kernel'].wrapping = 'clamp_to_edge'
        self.program['u_shape'] = I.shape[:2]
        self.show()

    def on_draw(self, event):
        gloo.clear(color=True, depth=True)
        self.program.draw('triangle_strip')

if __name__ == '__main__':
    c = Canvas()
    app.run()
