class_name PickupableComponent extends InteractableComponent

signal picked_up(pick_upper: Node)

func interact(interactor: Node):
	picked_up.emit(interactor)
