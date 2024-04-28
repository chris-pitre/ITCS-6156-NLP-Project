extends Node2D

@onready var zone: Area2D = $InteractZone 
var interact_object: TalkableComponent

func _process(delta):
	interact_object = null
	for entity in zone.get_overlapping_bodies():
		if entity.is_in_group("Talkable"):
			interact_object = entity.get_node("TalkableComponent") as TalkableComponent
			break

func _unhandled_input(event):
	if event.is_action_pressed("Interact") and interact_object:
		interact(interact_object)

func interact(object):
	object.start_chat(get_parent())
