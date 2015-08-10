#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np

from vispy.util.transforms import ortho
from vispy import gloo
from vispy import app
from vispy.util.filter import gaussian_filter
from vispy.io import _data_dir

# Image to be displayed
W, H = 100, 100
np.random.seed(1000)
noise = np.random.normal(size=(100, 100), loc=50, scale=150)
noise = gaussian_filter(noise, (4, 4, 0))
I = noise.astype(np.float32)
# normalize to [0., 1.]
I *= 1.0/I.max()

# A simple texture quad
data = np.zeros(4, dtype=[('a_position', np.float32, 2),
                          ('a_texcoord', np.float32, 2)])
data['a_position'] = np.array([[0, 0], [W, 0], [0, H], [W, H]])
data['a_texcoord'] = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])


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
#include "misc/spatial-filters.frag"
uniform vec2 u_shape;
uniform sampler2D u_texture;
varying vec2 v_texcoord;


//------ bicubic interpolation from scratch ------
// coefficients for the cubic polynomial
vec4 c0 = vec4(-1.0,  3.0, -3.0,  1.0 ) /  6.0;
vec4 c1 = vec4( 3.0, -6.0,  0.0,  4.0 ) /  6.0;
vec4 c2 = vec4(-3.0,  3.0,  3.0,  1.0 ) /  6.0;
vec4 c3 = vec4( 1.0,  0.0,  0.0,  0.0 ) /  6.0;

vec4 cubic(vec4 var, vec4 p0, vec4 p1, vec4 p2, vec4 p3 ) {
    // return cubic polynomial
    return  p0 * dot( c0, var) +
            p1 * dot( c1, var) +
            p2 * dot( c2, var) +
            p3 * dot( c3, var);
}

vec4 row(vec4 var, vec2 xy, vec2 shape, float pos) {
    // fetch texture at selected points
    vec4 p0 = texture2D(u_texture, (xy + vec2(-1, pos))/shape);
    vec4 p1 = texture2D(u_texture, (xy + vec2(0, pos))/shape);
    vec4 p2 = texture2D(u_texture, (xy + vec2(1, pos))/shape);
    vec4 p3 = texture2D(u_texture, (xy + vec2(2, pos))/shape);
    return cubic( var, p0, p1, p2, p3);
}

// bicubic interpolation
// see http://http.developer.nvidia.com/GPUGems2/gpugems2_chapter20.html
vec4 texture2D_bicubic(vec2 tc) {
    vec2 sh = u_shape;
    vec2 p = sh * tc - 0.5;
    vec2 a = fract(p);
    vec2 xy = floor(p);

    float x = a.x;
    float x2 = x * x;
    float x3 = x * x2;
    vec4 varx = vec4(x3, x2, x, 1.0);

    vec4 r0 = row( varx, xy, sh, -1);
    vec4 r1 = row( varx, xy, sh, 0);
    vec4 r2 = row( varx, xy, sh, 1);
    vec4 r3 = row( varx, xy, sh, 2);

    x = a.y;
    x2 = x * x;
    x3 = x * x2;
    vec4 vary = vec4(x3, x2, x, 1.0);

    return cubic( vary, r0, r1, r2, r3);
}
//------ bicubic interpolation from scratch ------

void main()
{
    // get texela using both interpolations
    vec4 texel1 = Bicubic(u_texture, u_shape, v_texcoord);
    vec4 texel2 = texture2D_bicubic(v_texcoord);

    // create contours from both texels
    float si1 = abs(ceil(texel1.r * 10.)/10.);
    vec4 col1 = vec4(si1,si1,si1,1.0);
    float si2 = abs(ceil(texel2.r * 10.)/10.);
    vec4 col2 = vec4(si2,si2,si2,1.0);


    // please comment out properly to see the three different
    // images
    // 1. subtract, should get all black image
    //gl_FragColor = abs((col1-col2));

    // 2. interpolation with 1D-kernel
    gl_FragColor = col1;

    // 3. interpolation from scratch
    //gl_FragColor = col2;
}

"""


class Canvas(app.Canvas):

    def __init__(self):
        app.Canvas.__init__(self, keys='interactive', size=((W * 5), (H * 5)))

        self.program = gloo.Program(VERT_SHADER, FRAG_SHADER)
        self.texture = gloo.Texture2D(I, interpolation='nearest')

        self.program['u_texture'] = self.texture
        self.program.bind(gloo.VertexBuffer(data))

        self.view = np.eye(4, dtype=np.float32)
        self.model = np.eye(4, dtype=np.float32)
        self.projection = np.eye(4, dtype=np.float32)

        self.program['u_model'] = self.model
        self.program['u_view'] = self.view
        self.projection = ortho(0, W, 0, H, -1, 1)
        self.program['u_projection'] = self.projection

        self.program['u_kernel'] = np.load(os.path.join(_data_dir, 'spatial-filters.npy'))
        self.program['u_kernel'].interpolation = 'nearest'
        self.program['u_shape'] = I.shape[1], I.shape[0]

        gloo.set_clear_color('white')

        self.show()

    def on_draw(self, event):
        gloo.clear(color=True, depth=True)
        self.program.draw('triangle_strip')


if __name__ == '__main__':
    c = Canvas()
    app.run()
