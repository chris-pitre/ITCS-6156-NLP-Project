class_name MovementComponent
extends Node2D

@export_category("References")
@export var character_body: CharacterBody2D
@export var input_component: InputComponent

@export_category("Settings")
@export var max_speed: int = 150
@export var acceleration: int = 1000
@export var friction: int = 600
@export var push_force: int = 100

var input := Vector2.ZERO

func _ready():
	if character_body == null:
		push_error("CharacterBody2D is not set for " + get_parent().name)

func _physics_process(delta):
	if character_body != null:
		do_movement(delta)
		
func do_movement(delta):
	if input_component != null:
		input = input_component.get_input_axis()
	else:
		push_error(name + " is missing movement input")
		
	if input == Vector2.ZERO:
		character_body.velocity = character_body.velocity.move_toward(Vector2.ZERO, friction * delta)
	else:
		character_body.velocity = character_body.velocity.move_toward(input * max_speed, acceleration * delta)
	character_body.move_and_slide()
	handle_collisions()

func handle_collisions():
	for i in character_body.get_slide_collision_count():
		var collision = character_body.get_slide_collision(i)
		var pushable_object = collision.get_collider() as RigidBody2D
		if pushable_object:
			pushable_object.apply_central_force(-collision.get_normal() * push_force)
