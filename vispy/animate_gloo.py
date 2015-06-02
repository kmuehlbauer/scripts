#!/usr/bin/env python
# -*- coding: utf-8 -*-
# vispy: gallery 2
"""
Example demonstrating showing a, image with a fixed ratio.
"""

import numpy as np

from vispy.util.transforms import ortho
from vispy import gloo
from vispy import app


# Image to be displayed
W, H = 360, 360

# A simple texture quad
data = np.zeros(4, dtype=[('a_position', np.float32, 2),
                          ('a_texcoord', np.float32, 2)])
data['a_position'] = np.array([[0, 0], [W, 0], [0, H], [W, H]])
data['a_texcoord'] = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])


VERT_SHADER = """
// Uniforms
uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_projection;
uniform float u_antialias;

// Attributes
attribute vec2 a_position;
attribute vec2 a_texcoord;

// Varyings
varying vec2 v_texcoord;

// Main
void main (void)
{
    v_texcoord = a_texcoord;
    gl_Position = u_projection * u_view * u_model * vec4(a_position,0.0,1.0);
}
"""

FRAG_SHADER = """
uniform sampler2D u_texture;
varying vec2 v_texcoord;

void main()
{
    /*vec2 uv = v_texcoord.xy;
    vec2 p = uv - 0.5;
    float r = length(p.xy);
    float a = atan(radians(p.y), radians(p.x));
    gl_FragColor = texture2D(u_texture, vec2(r, a));
    gl_FragColor = vec4(gl_FragColor.xyz / r , 1.0);*/

    gl_FragColor = texture2D(u_texture, v_texcoord);
    gl_FragColor.a = 1.0;
}

"""

class Canvas(app.Canvas):

    def __init__(self):
        app.Canvas.__init__(self, keys='interactive', size=((360), (360)))

        self.data = np.zeros((W, H)).astype(np.float32)
        for i in range(0,90):
            self.data[i,:] = i / 90.

        self.program = gloo.Program(VERT_SHADER, FRAG_SHADER)
        self.texture = gloo.Texture2D(self.data, interpolation='linear')
        self.program['u_texture'] = self.texture
        self.program.bind(gloo.VertexBuffer(data))

        self.view = np.eye(4, dtype=np.float32)
        self.model = np.eye(4, dtype=np.float32)
        self.projection = np.eye(4, dtype=np.float32)

        self.program['u_model'] = self.model
        self.program['u_view'] = self.view
        self.projection = ortho(0, W, 0, H, -1, 1)
        self.program['u_projection'] = self.projection

        gloo.set_clear_color('white')

        self._timer = app.Timer('auto', connect=self.update, start=True)
        self.measure_fps()

        self.show()

    def on_draw(self, event):
        gloo.clear(color=True, depth=True)
        self.data = np.roll(self.data, 1, axis=0)
        self.texture.set_data(self.data)
        self.program.draw('triangle_strip')

if __name__ == '__main__':
    c = Canvas()
    app.run()
