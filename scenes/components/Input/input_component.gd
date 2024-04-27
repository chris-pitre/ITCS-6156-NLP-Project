class_name InputComponent
extends Node2D

func get_input_axis() -> Vector2:
	var input := Vector2.ZERO
	
	input.x = Input.get_action_strength("ui_right") - Input.get_action_strength("ui_left")
	input.y = Input.get_action_strength("ui_down") - Input.get_action_strength("ui_up")
	
	input = input.normalized()
	
	return input
