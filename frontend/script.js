// API base URL - use relative path to work from any host
const API_URL = '/api';

// Global state
let currentSessionId = null;

// DOM elements
let chatMessages, chatInput, sendButton, totalCourses, courseTitles, newChatButton, themeToggle;

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    // Get DOM elements after page loads
    chatMessages = document.getElementById('chatMessages');
    chatInput = document.getElementById('chatInput');
    sendButton = document.getElementById('sendButton');
    totalCourses = document.getElementById('totalCourses');
    courseTitles = document.getElementById('courseTitles');
    newChatButton = document.getElementById('newChatButton');
    themeToggle = document.getElementById('themeToggle');
    
    setupEventListeners();
    initializeTheme();
    createNewSession();
    loadCourseStats();
});

// Event Listeners
function setupEventListeners() {
    // Chat functionality
    sendButton.addEventListener('click', sendMessage);
    chatInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') sendMessage();
    });
    
    // New chat button
    newChatButton.addEventListener('click', startNewChat);
    
    // Theme toggle button
    if (themeToggle) {
        themeToggle.addEventListener('click', toggleTheme);
    }
    
    // Suggested questions
    document.querySelectorAll('.suggested-item').forEach(button => {
        button.addEventListener('click', (e) => {
            const question = e.target.getAttribute('data-question');
            chatInput.value = question;
            sendMessage();
        });
    });
}


// Chat Functions
async function sendMessage() {
    const query = chatInput.value.trim();
    if (!query) return;

    // Disable input
    chatInput.value = '';
    chatInput.disabled = true;
    sendButton.disabled = true;

    // Add user message
    addMessage(query, 'user');

    // Add loading message - create a unique container for it
    const loadingMessage = createLoadingMessage();
    chatMessages.appendChild(loadingMessage);
    chatMessages.scrollTop = chatMessages.scrollHeight;

    try {
        const response = await fetch(`${API_URL}/query`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                query: query,
                session_id: currentSessionId
            })
        });

        if (!response.ok) throw new Error('Query failed');

        const data = await response.json();
        
        // Update session ID if new
        if (!currentSessionId) {
            currentSessionId = data.session_id;
        }

        // Replace loading message with response
        loadingMessage.remove();
        addMessage(data.answer, 'assistant', data.sources);

    } catch (error) {
        // Replace loading message with error
        loadingMessage.remove();
        addMessage(`Error: ${error.message}`, 'assistant');
    } finally {
        chatInput.disabled = false;
        sendButton.disabled = false;
        chatInput.focus();
    }
}

function createLoadingMessage() {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message assistant';
    messageDiv.innerHTML = `
        <div class="message-content">
            <div class="loading">
                <span></span>
                <span></span>
                <span></span>
            </div>
        </div>
    `;
    return messageDiv;
}

function addMessage(content, type, sources = null, isWelcome = false) {
    const messageId = Date.now();
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${type}${isWelcome ? ' welcome-message' : ''}`;
    messageDiv.id = `message-${messageId}`;
    
    // Convert markdown to HTML for assistant messages
    const displayContent = type === 'assistant' ? marked.parse(content) : escapeHtml(content);
    
    let html = `<div class="message-content">${displayContent}</div>`;
    
    if (sources && sources.length > 0) {
        console.log('Sources received:', sources); // Debug log
        
        // Create sources container
        const sourcesContainer = document.createElement('details');
        sourcesContainer.className = 'sources-collapsible';
        
        const summary = document.createElement('summary');
        summary.className = 'sources-header';
        summary.textContent = 'Sources';
        sourcesContainer.appendChild(summary);
        
        const sourcesContent = document.createElement('div');
        sourcesContent.className = 'sources-content';
        
        // Create source elements
        sources.forEach((source, index) => {
            console.log(`Source ${index}:`, source, 'Type:', typeof source); // Debug
            
            if (index > 0) {
                sourcesContent.appendChild(document.createTextNode(', '));
            }
            
            // Handle string format with embedded link (text|link)
            if (typeof source === 'string') {
                const parts = source.split('|');
                console.log(`Split parts:`, parts); // Debug
                if (parts.length === 2) {
                    // Has embedded link - create actual link element
                    const sourceText = parts[0];
                    const sourceLink = parts[1];
                    
                    console.log(`Creating link: "${sourceText}" -> "${sourceLink}"`); // Debug
                    
                    const linkElement = document.createElement('a');
                    linkElement.href = sourceLink;
                    linkElement.target = '_blank';
                    linkElement.rel = 'noopener noreferrer';
                    linkElement.className = 'source-link';
                    linkElement.textContent = sourceText;
                    linkElement.style.color = '#2563eb'; // Force blue color for testing
                    linkElement.style.textDecoration = 'underline'; // Force underline for testing
                    
                    console.log(`Created link element:`, linkElement); // Debug
                    console.log(`Link href:`, linkElement.href); // Debug
                    
                    sourcesContent.appendChild(linkElement);
                    console.log(`Appended link to sourcesContent`); // Debug
                } else {
                    // Plain text source
                    console.log(`Adding plain text source:`, source);
                    sourcesContent.appendChild(document.createTextNode(source));
                }
            } else {
                // Fallback for any other type
                console.warn('Unexpected source type:', typeof source, source);
                sourcesContent.appendChild(document.createTextNode(String(source)));
            }
        });
        
        sourcesContainer.appendChild(sourcesContent);
        
        // Don't convert to HTML - append directly
        messageDiv.innerHTML = html;
        messageDiv.appendChild(sourcesContainer);
        
        console.log('Appended sources container directly to messageDiv'); // Debug
    } else {
        messageDiv.innerHTML = html;
    }
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    
    return messageId;
}

// Helper function to escape HTML for user messages
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Removed removeMessage function - no longer needed since we handle loading differently

async function createNewSession() {
    currentSessionId = null;
    chatMessages.innerHTML = '';
    addMessage('Welcome to the Course Materials Assistant! I can help you with questions about courses, lessons and specific content. What would you like to know?', 'assistant', null, true);
}

function startNewChat() {
    // Provide visual feedback
    newChatButton.disabled = true;
    const originalText = newChatButton.textContent;
    newChatButton.textContent = 'STARTING...';
    
    // Clear current chat and start new session
    createNewSession();
    
    // Clear any text in the input
    if (chatInput) {
        chatInput.value = '';
        chatInput.focus();
    }
    
    // Reset button state
    setTimeout(() => {
        newChatButton.disabled = false;
        newChatButton.textContent = originalText;
    }, 300);
}

// Load course statistics
async function loadCourseStats() {
    try {
        console.log('Loading course stats...');
        const response = await fetch(`${API_URL}/courses`);
        if (!response.ok) throw new Error('Failed to load course stats');
        
        const data = await response.json();
        console.log('Course data received:', data);
        
        // Update stats in UI
        if (totalCourses) {
            totalCourses.textContent = data.total_courses;
        }
        
        // Update course titles
        if (courseTitles) {
            if (data.course_titles && data.course_titles.length > 0) {
                courseTitles.innerHTML = data.course_titles
                    .map(title => `<div class="course-title-item">${title}</div>`)
                    .join('');
            } else {
                courseTitles.innerHTML = '<span class="no-courses">No courses available</span>';
            }
        }
        
    } catch (error) {
        console.error('Error loading course stats:', error);
        // Set default values on error
        if (totalCourses) {
            totalCourses.textContent = '0';
        }
        if (courseTitles) {
            courseTitles.innerHTML = '<span class="error">Failed to load courses</span>';
        }
    }
}

// Theme Functions
function initializeTheme() {
    // Check for saved theme preference or default to 'dark'
    const savedTheme = localStorage.getItem('theme') || 'dark';
    setTheme(savedTheme);
}

function toggleTheme() {
    const currentTheme = document.documentElement.getAttribute('data-theme') || 'dark';
    const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
    setTheme(newTheme);
}

function setTheme(theme) {
    if (theme === 'light') {
        document.documentElement.setAttribute('data-theme', 'light');
    } else {
        document.documentElement.removeAttribute('data-theme');
    }
    
    // Save theme preference
    localStorage.setItem('theme', theme);
    
    // Add subtle feedback animation to toggle button
    if (themeToggle) {
        themeToggle.style.transform = 'scale(0.95)';
        setTimeout(() => {
            themeToggle.style.transform = '';
        }, 150);
    }
}