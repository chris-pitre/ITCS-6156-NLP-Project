extends StaticBody2D

@onready var sprite = $AnimatedSprite2D
@onready var collider = $CollisionShape2D

func _on_opened():
	sprite.play("open")
	collider.disabled = true
	
