extends Node

var udp := PacketPeerUDP.new()
var python_script_path:= "src/Python-NLP/server.py"
@export var show_terminal: bool
var pid: int

@export var text_entry: LimitedTextInput
@export var response_text: Label

func _ready():
	pid = OS.create_process("python", [python_script_path], show_terminal)
	udp.connect_to_host("127.0.0.1", 12345)

# Kill process when game is closed
func _notification(what):
	if what == NOTIFICATION_WM_CLOSE_REQUEST:
		send_message("<exit>")
		OS.kill(pid)

func _process(delta):
	if udp.get_available_packet_count() > 0:
		var response = udp.get_packet().get_string_from_utf8()
		print(response)
		response_text.text = response

func send_message(message: String):
	print("Sent message!")
	udp.put_packet(message.to_utf8_buffer())
	response_text.text = "..."
	
func _input(event):
	if event.is_action_pressed("ui_accept"): 
		send_message(text_entry.text)
		text_entry.clear_text()

