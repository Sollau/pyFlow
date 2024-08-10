import pyglet

window = pyglet.window.Window()

@window.event
def on_draw():
    window.clear()
    label = pyglet.text.Label('Hello, Pyglet!',
                              font_name='Times New Roman',
                              font_size=36,
                              x=window.width//2, y=window.height//2,
                              anchor_x='center', anchor_y='center')
    label.draw()

pyglet.app.run()
