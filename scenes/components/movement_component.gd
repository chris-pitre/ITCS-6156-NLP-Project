class_name MovementComponent
extends Node2D

@export_category("References")
@export var character_body: CharacterBody2D

@export_category("Settings")
@export var max_speed: int = 150
@export var acceleration: int = 1000
@export var friction: int = 600

var input := Vector2.ZERO

func _ready():
	if character_body == null:
		push_error("CharacterBody2D is not set for " + get_parent().name)

func _physics_process(delta):
	if character_body != null:
		do_movement(delta)
		
func do_movement(delta):
	input.x = Input.get_action_strength("ui_right") - Input.get_action_strength("ui_left")
	input.y = Input.get_action_strength("ui_down") - Input.get_action_strength("ui_up")
	
	input = input.normalized()
	
	if input == Vector2.ZERO:
		character_body.velocity = character_body.velocity.move_toward(Vector2.ZERO, friction * delta)
	else:
		character_body.velocity = character_body.velocity.move_toward(input * max_speed, acceleration * delta)
		
	character_body.move_and_slide()
	
