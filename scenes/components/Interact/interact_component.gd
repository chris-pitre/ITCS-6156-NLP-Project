extends Node2D

@onready var zone: Area2D = $InteractZone 
var interact_object: InteractableComponent

func _process(delta):
	interact_object = null
	for entity in zone.get_overlapping_bodies():
		for node in entity.get_children():
			if node is InteractableComponent:
				print(node)
				interact_object = node as InteractableComponent
				break

func _unhandled_input(event):
	if event.is_action_pressed("Interact") and interact_object:
		interact(interact_object)

func interact(object):
	object.interact(get_parent())
