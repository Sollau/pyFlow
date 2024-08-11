import pyglet

# window
window = pyglet.window.Window(resizable=True)

# graphics
batch = pyglet.graphics.Batch()

# bird
bird = pyglet.shapes.Circle(500, 500, 30, color = (255, 0, 0,), batch=batch )
bird.visible = True

# movement
def on_key_press(symbol, modifiers):
    if symbol == pyglet.window.key.UP:
        print(f"Key UP pressed")
        bird.y +=20
        pass

    if symbol == pyglet.window.key.DOWN:
        print(f"Key DOWN pressed")
        bird.y-=20
        pass

    if modifiers:
        pass


def on_key_release(symbol, modifiers):
    if symbol:
        pass
    if modifiers:
        pass


window.push_handlers(on_key_press, on_key_release)


# draw
@window.event
def on_draw():
    window.clear()
    batch.draw()


# run
pyglet.app.run()
