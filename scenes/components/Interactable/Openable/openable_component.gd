class_name OpenableComponent extends InteractableComponent

@export var open: bool = false
signal opened


func interact(interactor: Node):
	open = true
	opened.emit()
