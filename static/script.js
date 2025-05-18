const API_BASE_URL = 'http://localhost:5001'; // Your Flask backend URL

let selectedCharacterId = null;
let selectedChatTargetId = null; // For chat
let allCharacters = []; // Store all characters for easy access

async function fetchWorldState() {
    try {
        const response = await fetch(`${API_BASE_URL}/world_state`);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const worldState = await response.json();
        if (worldState && worldState.characters) {
            allCharacters = worldState.characters; // Cache characters
        }
        return worldState;
    } catch (error) {
        console.error("Failed to fetch world state:", error);
        document.getElementById('world-info').textContent = 'Error loading world state. Is the backend running?';
        return null;
    }
}

function renderBuilding(buildingData, container) {
    container.innerHTML = ''; // Clear previous state

    if (!buildingData) {
        container.textContent = 'No building data to render.';
        return;
    }

    const buildingElement = document.createElement('div');
    buildingElement.className = 'building';
    
    const buildingNameElement = document.createElement('h2');
    buildingNameElement.textContent = buildingData.name || 'Unnamed Building';
    buildingElement.appendChild(buildingNameElement);

    const gridElement = document.createElement('div');
    gridElement.className = 'building-grid';
    
    const [width, height] = buildingData.internal_dims;
    gridElement.style.gridTemplateColumns = `repeat(${width}, 1fr)`;
    gridElement.style.gridTemplateRows = `repeat(${height}, 1fr)`;

    // tiles_grid is [row][col] which means [y][x]
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const tileCell = document.createElement('div');
            tileCell.className = 'tile';
            tileCell.dataset.x = x;
            tileCell.dataset.y = y;

            const tileData = buildingData.tiles_grid[y][x];
            if (tileData && tileData.occupants && tileData.occupants.length > 0) {
                // For simplicity, render the first occupant
                // In your data, occupant.render is the emoji/char
                tileCell.textContent = tileData.occupants[0].render; 
                tileCell.classList.add('occupied');
                tileCell.title = tileData.occupants.map(o => `${o.name} (${o.type})`).join(', ');
            } else {
                tileCell.textContent = '.'; // Default for empty tile
            }
            gridElement.appendChild(tileCell);
        }
    }
    buildingElement.appendChild(gridElement);
    container.appendChild(buildingElement);
}

function updateWorldInfo(worldState) {
    const infoElement = document.getElementById('world-info');
    if (worldState) {
        infoElement.textContent = `World Time: ${worldState.world_time}`;
    }
}

function populateCharacterDropdown(characters) {
    const dropdown = document.getElementById('character-select-dropdown');
    dropdown.innerHTML = '<option value="">--Select a Character--</option>';

    if (characters && characters.length > 0) {
        characters.forEach(character => {
            const option = document.createElement('option');
            option.value = character.id;
            option.textContent = `${character.name} (${character.race})`;
            dropdown.appendChild(option);
        });
    }

    dropdown.addEventListener('change', (event) => {
        selectedCharacterId = event.target.value;
        const selectedCharacterNameDisplay = document.getElementById('selected-character-name');
        if (selectedCharacterId) {
            const selectedChar = characters.find(c => c.id === selectedCharacterId);
            selectedCharacterNameDisplay.textContent = selectedChar ? selectedChar.name : 'None';
            populateChatTargetDropdown(allCharacters, selectedCharacterId); // Update chat targets
        } else {
            selectedCharacterNameDisplay.textContent = 'None';
            populateChatTargetDropdown([], null); // Clear chat targets if no char selected
        }
        console.log("Selected character ID:", selectedCharacterId);
    });
}

function populateChatTargetDropdown(characters, currentPlayerId) {
    const dropdown = document.getElementById('chat-target-select');
    dropdown.innerHTML = '<option value="">--Select Target--</option>'; // Clear existing options
    selectedChatTargetId = null; // Reset target selection

    if (characters && characters.length > 0 && currentPlayerId) {
        characters.forEach(character => {
            if (character.id !== currentPlayerId) { // Don't allow chatting with oneself
                const option = document.createElement('option');
                option.value = character.id;
                option.textContent = `${character.name} (${character.race})`;
                dropdown.appendChild(option);
            }
        });
    }

    dropdown.addEventListener('change', (event) => {
        selectedChatTargetId = event.target.value;
        console.log("Selected chat target ID:", selectedChatTargetId);
    });
}

function addMessageToChatLog(sourceName, targetName, message, isSystem = false, success = true) {
    const chatLog = document.getElementById('chat-log');
    const messageElement = document.createElement('p');
    
    if (isSystem) {
        messageElement.innerHTML = `<em>${message}</em>`;
        if (!success) messageElement.style.color = 'red';
    } else {
        messageElement.textContent = `${sourceName} to ${targetName}: ${message}`;
    }
    
    // If it's the placeholder, remove it
    const placeholder = chatLog.querySelector('em');
    if (placeholder && placeholder.textContent.includes("Chat messages will appear here")) {
        chatLog.innerHTML = ''; 
    }

    chatLog.appendChild(messageElement);
    chatLog.scrollTop = chatLog.scrollHeight; // Scroll to the bottom
}

async function sendChatMessage() {
    const messageInput = document.getElementById('chat-message-input');
    const message = messageInput.value.trim();

    if (!selectedCharacterId) {
        addMessageToChatLog(null, null, "You must select your character first.", true, false);
        return;
    }
    if (!selectedChatTargetId) {
        addMessageToChatLog(null, null, "You must select a target character to chat with.", true, false);
        return;
    }
    if (!message) {
        addMessageToChatLog(null, null, "You cannot send an empty message.", true, false);
        return;
    }

    const sourceChar = allCharacters.find(c => c.id === selectedCharacterId);
    const targetChar = allCharacters.find(c => c.id === selectedChatTargetId);

    if (!sourceChar || !targetChar) {
        addMessageToChatLog(null, null, "Error finding source or target character.", true, false);
        return;
    }

    console.log(`Sending chat from ${sourceChar.name} (ID: ${selectedCharacterId}) to ${targetChar.name} (ID: ${selectedChatTargetId}): ${message}`);

    try {
        const response = await fetch(`${API_BASE_URL}/action/chat`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                source_character_id: selectedCharacterId,
                target_character_id: selectedChatTargetId,
                message: message,
            }),
        });
        const result = await response.json();
        if (result.success) {
            addMessageToChatLog(sourceChar.name, targetChar.name, message);
            messageInput.value = ''; // Clear input field on success
            await refreshWorldView(); // Refresh world (e.g. time might have passed)
        } else {
            addMessageToChatLog(sourceChar.name, targetChar.name, `Chat failed: ${result.message}`, true, false);
            console.error("Chat failed:", result.message);
        }
    } catch (error) {
        addMessageToChatLog(sourceChar.name, targetChar.name, `Error sending chat: ${error}`, true, false);
        console.error("Error sending chat message:", error);
    }
}

async function moveCharacter(characterId, direction) {
    if (!characterId) {
        console.warn("No character selected to move.");
        return;
    }
    console.log(`Attempting to move character ${characterId} ${direction}`);
    try {
        const response = await fetch(`${API_BASE_URL}/action/move`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                character_id: characterId,
                direction: direction.toUpperCase(),
            }),
        });
        const result = await response.json();
        if (result.success) {
            console.log(result.message);
            await refreshWorldView(); // Refresh the view to show movement
        } else {
            console.error("Move failed:", result.message);
            alert(`Move failed: ${result.message}`);
        }
    } catch (error) {
        console.error("Error moving character:", error);
        alert("Error moving character. Check console.");
    }
}

function handleKeyPress(event) {
    if (!selectedCharacterId) {
        // console.log("No character selected. Ignoring key press for movement.");
        return; 
    }

    let direction = null;
    switch (event.key.toUpperCase()) {
        case 'W':
            direction = 'NORTH';
            break;
        case 'A':
            direction = 'WEST';
            break;
        case 'S':
            direction = 'SOUTH';
            break;
        case 'D':
            direction = 'EAST';
            break;
        default:
            return; // Not a movement key
    }
    event.preventDefault(); // Prevent default browser action for these keys (e.g., scrolling)
    moveCharacter(selectedCharacterId, direction);
}

async function refreshWorldView() {
    const worldState = await fetchWorldState();
    const gameContainer = document.getElementById('game-container');
    
    updateWorldInfo(worldState);
    // populateCharacterDropdown(worldState.characters); // Refresh dropdown if characters can change dynamically

    if (worldState && worldState.buildings && worldState.buildings.length > 0) {
        // For now, render the first building. 
        // Consider if the selected character is in a different building or how to handle multiple buildings.
        renderBuilding(worldState.buildings[0], gameContainer);
    } else {
        gameContainer.textContent = 'No buildings found in the world state.';
         if (worldState && !worldState.buildings) {
            console.error("worldState.buildings is undefined or null", worldState);
        }
    }
}

async function main() {
    const worldState = await fetchWorldState();
    const gameContainer = document.getElementById('game-container');
    
    updateWorldInfo(worldState);

    if (worldState && worldState.characters) {
        // allCharacters is already set by fetchWorldState
        populateCharacterDropdown(allCharacters);
        // Initialize chat target dropdown (it will be empty until a character is selected)
        populateChatTargetDropdown(allCharacters, null);
    } else {
        console.warn("No characters found to populate dropdowns.");
    }

    if (worldState && worldState.buildings && worldState.buildings.length > 0) {
        // For now, render the first building
        renderBuilding(worldState.buildings[0], gameContainer);
    } else {
        gameContainer.textContent = 'No buildings found in the world state.';
         if (worldState && !worldState.buildings) {
            console.error("worldState.buildings is undefined or null", worldState);
        }
    }

    document.addEventListener('keydown', handleKeyPress);
    document.getElementById('send-chat-button').addEventListener('click', sendChatMessage);
}

// Initial load
main();

// TODO: Periodically refresh world state or use WebSockets for real-time updates
// TODO: Better handling for multiple buildings if player can move between them
// TODO: Visual feedback for selected character on the grid
// TODO: Consider dynamically updating chat targets based on adjacency after moves. 