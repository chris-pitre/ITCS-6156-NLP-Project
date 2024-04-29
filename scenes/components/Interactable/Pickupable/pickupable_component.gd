class_name PickupableComponent extends InteractableComponent

@export var collider: CollisionShape2D

func interact(interactor: Node):
	get_parent().reparent(interactor)
	if collider:
		collider.disabled = true
