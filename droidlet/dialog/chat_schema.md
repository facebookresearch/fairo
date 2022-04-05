## Description ##
This is the source of truth for the most updated version of the schema to be followed when sending chats from the agent the the user (and perhaps eventually from the agent to other agents).  When updating this document please increment the version number.  Older versions of the schema can be found in git history.

Note that chats sent from the user to the agent should always be in plaintext, as all messages should be parsed by the NSP and interpreted.  I.e. special pathways and conditionals on agent input are discouraged.  It is possible to help the NSP by including special tokens that are reserved for specific use cases, eg. clarification response.

### Content Types ###
 - `chat_string`: Only text to display in the chat window with no response assumed.
    - Example: "I finished building this!"
 - `chat_and_media`: Text and media to display in the chat window, with no response assumed.
    - Example: "I finished building this!" (with accompanying photo)
 - `chat_and_text_options`: Text to display in the chat window along with a list of possible response options.
    - Example: "Are these the Droidlets you are looking for?" ('no', 'no')
 - `chat_and_media_options`: Text to display in the chat window along with a list of clickable media options.
    - Example: "Which of these is the 'blue sphere' you were referring to?" (pictures of blue spheres)
 - `chat_and_media_and_text_options`: Text and media to display in the chat window alond with a list of possible response options.
    - Example: "Is this funny?" <img src='https://external-preview.redd.it/hWh_8TpqrT6zAwpzHJ_m9Rx3iHjc_yI4zSI6aazMFTc.jpg?auto=webp&s=6d8006ac3edca5bad98dd7b5b9a4a8d5554eaff0' width="200"> ('yes', 'no')

### Schema ###

<pre>
{
	"title": "Droidlet Chat",
	"description": "A single chat sent from a Droidlet agent to the user",
    "version": 1,
	"type": "object",
	
	"properties": {
		"speaker_id": {
			"description": "The memid of the speaker",
			"type": "string"
		},

		"timestamp": {
			"description": "Unix timestamp of when the chat was sent",
			"type": "integer"
		},

		"content_type": {
			"description": "The type of content to be rendered to the user",
			"type": "array",
			"items": {
                "enum": [
                    "chat_string",
                    "chat_and_media",
                    "chat_and_text_options",
                    "chat_and_media_options",
                    "chat_and_media_and_text_options"
                ]
			},
            "minItems": 1,
            "maxItems": 1
		},
		
		"content": {
			"description": "The chat content",
			"type": "array",
            "items": {
                "type": "array",
                "items": [
                    {"enum": ["text", "image_link", "response_option", "response_image_link"]},
                    {"type": "string"}
                ],
                "minItems": 2,
                "maxItems": 2
			},
            "minItems": 1,
		},

	"required": ["speaker_id", "timestamp", "content_type", "content"]
}
</pre>
