class_name OpenableComponent extends InteractableComponent

signal open_attempt(opener: Node)


func interact(interactor: Node):
	open_attempt.emit(interactor)
