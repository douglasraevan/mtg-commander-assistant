# MTG Commander Assistant

An AI-powered assistant for Magic: The Gathering Commander deck analysis and discussion. This application combines the power of LangChain, OpenAI's GPT models, and the Scryfall API to provide intelligent insights about your Commander decks.

## Features

- 🤖 AI-powered card analysis and deck discussion
- 📁 Support for deck list file uploads
- 🔍 Automatic card data fetching from Scryfall API
- 💬 Interactive chat interface
- 🔄 Real-time streaming responses
- 🎴 Comprehensive card information including mana cost, type, and oracle text

## Prerequisites

- Python 3.10+
- Poetry
- OpenAI API key
- Internet connection (for Scryfall API access)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/mtg-commander-assistant.git
cd mtg-commander-assistant
```

2. Install required dependencies:
```bash
poetry install
```

3. Create a `.env` file in the project root and add your OpenAI API key:
```bash
OPENAI_API_KEY=your_api_key_here
```

## Usage

1. Start the application:
```bash
poetry run python -m project
```

2. Open your web browser and navigate to the provided local URL (typically http://localhost:7860)

3. You can interact with the assistant in two ways:
   - Upload a deck list text file using the "Upload Text File" button
   - Type questions directly in the chat interface about your deck or Magic: The Gathering in general

### Deck List Format

Your deck list should be in a text file with the following format:
```
1 Card Name
1 Another Card
2 Multiple Copies Card
```

The last card in the list will be considered the Commander.

## Features in Detail

### Card Data Processing
The application automatically processes your deck list and fetches detailed card information from Scryfall, including:
- Card name
- Mana cost
- Type line
- Oracle text

### AI Assistant
The chat interface provides:
- Intelligent card analysis
- Deck building suggestions
- Rules clarifications
- Strategy discussions

## Technical Details

The project uses several key technologies:
- LangChain for AI chain management
- Gradio for the web interface
- OpenAI's GPT models for intelligent responses
- Scryfall API for card data
- LangGraph for workflow management

## Rate Limiting

The application respects Scryfall's API rate limits with a 100ms delay between requests to ensure reliable operation.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[TODO]

## Acknowledgments

- Scryfall API for providing comprehensive MTG card data
- OpenAI for their GPT models
- The MTG community for inspiration and support
