extends Area2D

@export var win_scene: PackedScene

func _on_body_entered(body):
	if body.is_in_group("Player"):
		get_tree().change_scene_to_packed(win_scene)
