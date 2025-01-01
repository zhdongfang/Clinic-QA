<template>
    <div class="chat-container">
        <h1>医疗智能问答机器人</h1>
        <div class="chat-window">
            <div v-for="message in messages" :key="message.id" class="message" :class="message.role">
                <img v-if="message.role === 'user'" src="@/assets/user.png" alt="" class="avatar user" />
                <img v-else src="@/assets/robot.png" alt="" class="avatar robot" />
                <div class="message-content">
                    <p>{{ message.content }}</p>
                </div>
            </div>
        </div>
        <div class="input-area">
            <input v-model="query" @keyup.enter="sendMessage" placeholder="请输入您的问题..." />
            <button @click="sendMessage">发送</button>
        </div>
        <div class="sidebar">
            <div class="settings-container">
                <div class="setting-item model-selection">
                    <label for="model">选择模型:</label>
                    <select v-model="selected_model" id="model">
                        <option value="Qwen2.5-7B">Qwen2.5-7B</option>
                        <option value="ChatGLM3-6B">ChatGLM3-6B</option>
                        <option value="Llama3.1-8B-Chinese-Chat">Llama3.1-8B-Chinese-Chat</option>
                    </select>
                </div>
            </div>
        </div>
    </div>
</template>

<script>
    import axios from 'axios'

    export default {
        name: 'ChatRobot',
        data() {
            return {
                query: '',
                messages: [],
                showEnt: false,
                showInt: false,
                showPrompt: false,
                selected_model: 'Llama3.1-8B-Chinese-Chatt', 
            }
        },
        methods: {
            async sendMessage() {
                if (this.query.trim() === '') return

                this.messages.push({
                    id: Date.now(),
                    role: 'user',
                    content: this.query,
                })

                try {
                    const response = await axios.post('http://localhost:8000/api/chat', {
                        query: this.query,
                        selected_model: this.selected_model,
                    })

                    this.messages.push({
                        id: Date.now(),
                        role: 'assistant',
                        content: response.data.answer,
                        details: {
                            entities: response.data.entities,
                            intent: response.data.intent,
                            prompt: response.data.prompt,
                        },
                    })
                } catch (error) {
                    console.error(error)
                    this.messages.push({
                        id: Date.now(),
                        role: 'assistant',
                        content: 'Sorry, we are unable to process your request.',
                    })
                }

                this.query = ''
            },
        },
    }
</script>

<style scoped>
    .chat-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        font-family: Avenir, Helvetica, Arial, sans-serif;
        padding: 20px;
    }

    h1 {
        margin-bottom: 20px;
    }

    .chat-window {
        border: none;
        padding: 10px;
        width: 80%;
        height: 650px;
        overflow-y: scroll;
        margin-bottom: 20px;
        background-color: #f9f9f9;
    }

    .message {
        display: flex;
        align-items: flex-start;
        margin-bottom: 15px;
    }

    .avatar {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        margin-right: 10px;
    }

    .message-content {
        background-color: #e0f7fa;
        padding: 10px 15px;
        border-radius: 10px;
        max-width: 70%;
        font-size: 16px;
    }

    .message.assistant .message-content {
        background-color: #ffecb3;
    }

    .input-area {
        display: flex;
        width: 80%;
        margin-bottom: 20px;
    }

    .input-area input {
        flex: 1;
        padding: 10px;
        font-size: 16px;
        border: 1px solid #ccc;
        border-radius: 4px;
    }

    .input-area button {
        padding: 10px 20px;
        font-size: 16px;
        margin-left: 10px;
        border: none;
        background-color: #42b983;
        color: white;
        border-radius: 4px;
        cursor: pointer;
    }

    .input-area button:hover {
        background-color: #369870;
    }

    .sidebar {
        border: none;
        width: 80%;
        padding: 10px;
    }

    .sidebar h2 {
        margin-bottom: 10px;
    }

    .settings-container {
        display: flex;
        flex-wrap: wrap;
        justify-content: center;
        align-items: center;
    }

    .setting-item {
        display: flex;
        align-items: center;
        margin: 5px 10px;
    }

    .model-selection {
        display: flex;
        align-items: center;
    }

    .model-selection label {
        margin-right: 10px;
    }

    .model-selection select {
        padding: 5px;
        font-size: 16px;
    }
</style>