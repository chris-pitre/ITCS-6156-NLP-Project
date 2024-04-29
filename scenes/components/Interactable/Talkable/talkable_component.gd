class_name TalkableComponent
extends InteractableComponent

@onready var chat_box: LineEdit = $TextEntry/LineEdit
var activated: bool = false
var chatter: Node

func interact(interactor: Node):
	visible = true
	chat_box.grab_focus()
	chatter = interactor
	chatter.process_mode = Node.PROCESS_MODE_DISABLED
	activated = true

func end_chat():
	visible = false
	chat_box.release_focus()
	chatter.process_mode = Node.PROCESS_MODE_INHERIT
	activated = false
	
func _input(event):
	if event.is_action_pressed("ui_cancel") and activated:
		end_chat()
