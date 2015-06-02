
import numpy as np
from vispy import app
from vispy import gloo
from vispy import visuals
from vispy.visuals.transforms import STTransform, TransformSystem, BaseTransform, AffineTransform

class PolarTransform(BaseTransform):
    """Polar transform

    Maps (theta, r, z) to (x, y, z), where `x = r*cos(theta)`
    and `y = r*sin(theta)`.
    """
    glsl_map = """
        vec4 polar_transform_map(vec4 pos) {
            //return vec4(pos.y * cos(pos.x), pos.y * sin(pos.x), pos.z, 1);
            return vec4(pos.x * cos(pos.y), pos.x * sin(pos.y), pos.z, 1);
        }
        """

    glsl_imap = """
        vec4 polar_transform_map(vec4 pos) {
            // TODO: need some modulo math to handle larger theta values..?
            float theta = atan(radians(pos.y), radians(pos.x));
            theta = degrees(theta + 3.14159265358979323846);
            float r = length(pos.xy);
            return vec4(r, theta, pos.z, 1);
        }
        """

    Linear = False
    Orthogonal = False
    NonScaling = False
    Isometric = False

class Canvas(app.Canvas):
    def __init__(self):
        app.Canvas.__init__(self, keys=None, size=(360, 360))

        self.data = np.zeros((360, 360))

        for i in range(0,90):
            self.data[i,:] = i * 90

        self.image = visuals.ImageVisual(self.data, method='auto', cmap='cubehelix', clim='auto')
        self.image.tr_sys = TransformSystem(self)

        #tr = AffineTransform()
        #tr.rotate(90, (0, 0, 1))
        #self.image.transform = (STTransform(scale=(3.5,3.5), translate=(450, 450, 1)) *
        #                        tr *
        #                        PolarTransform())
        #self.image.tr_sys.visual_to_document = self.image.transform

        self._timer = app.Timer(start=False)
        self._timer.connect(self.update)
        self._timer.start()
        self.show()
        self.measure_fps()

    def on_resize(self,e):
        width, height = e.size[0], e.size[1]
        gloo.set_viewport(0, 0, width, height)

    def on_draw(self, ev):
        gloo.clear(color='white', depth=True)
        gloo.set_viewport(0, 0, *self.physical_size)
        self.data = np.roll(self.data, 1, axis=0)
        self.image.set_data(self.data)
        self.image.draw(self.image.tr_sys)

if __name__ == '__main__':
    c = Canvas()
    app.run()
