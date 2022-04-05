
<pre>
{
	"title": "Droidlet Chat",
	"description": "A single chat sent to or from a Droidlet agent",
	"type": "object",
	
	"properties": {
		"id": {
			"description": "The chat unique identifier",
			"type": "string"
		},

		"timestamp": {
			"description": "Unix timestamp of when the chat was sent",
			"type": "integer"
		},

		"class": {
			"description": "The class(s) indiciating the purpose of the chat",
			"type": "array",
			"items": {
					"type": "string",
					"enum": ["command", "noop", "clarification", "clarification_response"]
			}
		},
		
		"content": {
			"description": "The chat text",
			"type": "string"
		},
		
		"media": {
			"description": "Any media accompanying the chat",
			"type": "string",
			"encoding": {
				"type": "array",
				"items": {
						"type": "string",
						"enum": ["base64"]
				}
			}
		}
	},

	"required": ["id", "timestamp", "class", "content"]
}
</pre>
