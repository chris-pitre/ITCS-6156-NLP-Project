class_name TalkableComponent
extends Control

@onready var chat_box: LineEdit = $TextEntry/LineEdit
var activated: bool = false
var chatter: Node

func start_chat(player):
	visible = true
	chat_box.grab_focus()
	chatter = player
	chatter.process_mode = Node.PROCESS_MODE_DISABLED

func end_chat():
	visible = false
	chat_box.release_focus()
	chatter.process_mode = Node.PROCESS_MODE_INHERIT
	
func _input(event):
	if event.is_action_pressed("ui_cancel"):
		end_chat()
