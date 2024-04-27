class_name TalkableComponent
extends Control

@onready var chat_box: LineEdit = $TextEntry/LineEdit

func start_chat():
	visible = true
	chat_box.grab_focus()

func end_chat():
	visible = false
	chat_box.release_focus()
	
