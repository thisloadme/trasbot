class Chatbox {
    constructor() {
        this.args = {
            openButton: document.querySelector('.button_chatbox'),
            chatBox: document.querySelector('.isi_chatbox'),
            sendButton: document.querySelector('.button_send'),
        }

        this.state = false;
        this.messages = [];
    }

    display() {
        const {openButton, chatBox, sendButton} = this.args;

        openButton.addEventListener('click', () => this.toggleState(chatBox))

        sendButton.addEventListener('click', () => this.onSendButton(chatBox))

        const node = chatBox.querySelector('input');
        node.addEventListener('keyup', ({key}) => {
            if (key === "Enter") {
                this.onSendButton(chatBox)
            }
        })
    }

    toggleState(chatbox) {
        this.state = !this.state

        if (this.state) {
            chatbox.classList.add('chatbox_active')
        } else {
            chatbox.classList.remove('chatbox_active')
        }
    }

    onSendButton(chatbox) {
        var textField = chatbox.querySelector('input')
        let text1 = textField.value
        if (text1 === '') {
            return
        }

        let msg1 = {name: 'User', message: text1}
        this.messages.push(msg1)

        fetch('http://127.0.0.1:3000//chat', {
            method: 'POST',
            body: JSON.stringify({message: text1}),
            mode: 'cors',
            headers: {'Content-Type': 'application/json'}
        })
        .then(r => r.json())
        .then(r => {
            let msg2 = {name: 'Trasbot', message: r.answer}
            this.messages.push(msg2)
            this.updateChatText(chatbox)
            textField.value = ''
        }).catch(err => {
            console.error('Error: ', err)
            this.updateChatText(chatbox)
            textField.value = ''
        });
    }

    updateChatText(chatbox) {
        var html = ''
        this.messages.slice().reverse().forEach(function (item) {
            if (item.name === 'Trasbot') {
                html += '<div class="item_messages item_messages_visitor">' + item.message + '</div>'
            } else {
                html += '<div class="item_messages item_messages_operator">' + item.message + '</div>'
            }
        })

        const chatMessage = chatbox.querySelector('.messages_chatbox')
        chatMessage.innerHTML = html
    }
}

const chatbox = new Chatbox()
chatbox.display()