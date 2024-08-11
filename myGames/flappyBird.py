import pyglet
import random

# window
window = pyglet.window.Window(1200, 800)

# graphics
batch = pyglet.graphics.Batch()

# run distance
run_dist = 0
distance = pyglet.text.Label("Distance: " + str(run_dist), 10, 775, batch=batch)


# bird
xb = 200
yb = 400
delta_x = 50
delta_y = 25
bird = pyglet.shapes.Triangle(xb, yb, xb-delta_x, yb+delta_y, xb-delta_x, yb-delta_y, color = (255, 255, 0), batch=batch)


# bird movement
def on_key_press(symbol, modifiers):
    if symbol == pyglet.window.key.UP:
        if bird.y>775:
            bird.y = 25
        else:
            bird.y +=20
        pass

    if symbol == pyglet.window.key.DOWN:
        if bird.y<25:
            bird.y = 775
        else:
            bird.y -=20
        pass
    
    if symbol == pyglet.window.key.Q:
        window.close()
        pass
        
    if modifiers:
        pass

window.push_handlers(on_key_press)

# obstacles
obstacles = []

while len(obstacles)<10:
    y =random.uniform(0, 800)
    x = random.uniform(xb, 1200)
    obstacle = pyglet.shapes.Circle(x, y, 20, color = (255, 0, 0,), batch=batch)
    obstacles.append(obstacle)

def move_obstacles(dt):
    for i in range(0, len(obstacles)):
        obstacles[i].x -= 120*dt
        if obstacles[i].x < 20:
            obstacles[i].x = 1180
            obstacles[i].y = random.uniform(0, 800)
    global run_dist
    run_dist+=60*dt
    distance.text = "Distance: " + str(int(run_dist))
    
pyglet.clock.schedule_interval(move_obstacles, 1/60)


# draw
@window.event
def on_draw():
    window.clear()
    batch.draw()


# run
pyglet.app.run()
