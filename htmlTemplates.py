css = '''
<style>
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        align-items: flex-start;
    }

    .chat-message.user {
        #background-color: #f0f0f0;
        background-color: #00008B;
    }

    .chat-message.bot {
        background-color: #007bff;
        color: #fff;
    }

    .chat-message .avatar {
        width: 20%;
    }

    .chat-message .avatar img {
        max-width: 50px;
        max-height: 50px;
        border-radius: 50%;
        object-fit: cover;
    }

    .chat-message .message {
        width: 80%;
        padding: 0 1.5rem;
    }
</style>
'''

bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://cdn3d.iconscout.com/3d/premium/thumb/adobe-illustrator-6347386-5588278.png?f=webp">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="https://static.vecteezy.com/system/resources/thumbnails/016/774/644/small_2x/3d-user-icon-on-transparent-background-free-png.png">
    </div>    
    <div class="message">{{MSG}}</div>
</div>
'''
