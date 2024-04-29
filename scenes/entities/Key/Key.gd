extends Node2D

func _on_picked_up(interactor):
	interactor.has_key = true
