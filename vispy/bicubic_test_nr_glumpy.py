#! /usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from glumpy import app, gl, gloo, data, library

vertex = """
    attribute vec2 position;
    attribute vec2 texcoord;
    varying vec2 v_texcoord;
    void main()
    {
        gl_Position = vec4(position, 0.0, 1.0);
        v_texcoord = texcoord;
    } """

fragment = """
#include "misc/spatial-filters.frag"

uniform sampler2D u_data;
uniform vec2 u_shape;
varying vec2 v_texcoord;
void main()
{
    //gl_FragColor = Bicubic(u_data, u_shape, v_texcoord);
    const float level = 10.0;
    vec4 texel1 = Bicubic(u_data, u_shape, v_texcoord);

    float si1 = abs(ceil(texel1.r * level)/level);
    vec4 col1 = vec4(si1,si1,si1,1.0);
    gl_FragColor = col1;

} """

window = app.Window(width=512, height=512)

@window.event
def on_draw(dt):
    window.clear()
    program.draw(gl.GL_TRIANGLE_STRIP)

np.random.seed(1000)
program = gloo.Program(vertex, fragment, count=4)
program["position"] = (-1,-1), (-1,+1), (+1,-1), (+1,+1)
program['texcoord'] = ( 0, 0), ( 0,+1), (+1, 0), (+1,+1)
program['u_data'] = np.random.uniform(0,1,(5,5)).astype(np.float32)
program['u_shape'] = program['u_data'].shape[:2]
program['u_kernel'] = data.get("spatial-filters.npy")
program['u_kernel'].interpolation = gl.GL_NEAREST
program['u_kernel'].wrapping = gl.GL_CLAMP
program['u_data'].interpolation = gl.GL_NEAREST
program['u_data'].wrapping = gl.GL_CLAMP

app.run()