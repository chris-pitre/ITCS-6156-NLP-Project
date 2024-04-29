extends StaticBody2D

@onready var sprite = $AnimatedSprite2D
@onready var collider = $CollisionShape2D

func open():
	sprite.play("open")
	collider.disabled = true

func _on_open_attempt(opener: Node):
	for node in opener.get_children():
		if node.is_in_group("Key"):
			open()
			break
