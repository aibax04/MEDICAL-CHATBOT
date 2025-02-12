$(document).ready(function () {
    $("#send-button").click(function () {
        sendMessage();
    });

    $("#user-input").keypress(function (event) {
        if (event.which === 13) {
            sendMessage();
        }
    });

    function sendMessage() {
        let userMessage = $("#user-input").val();
        if (userMessage.trim() === "") return;

        let timestamp = new Date().toLocaleTimeString();

        $("#chat-box").append(
            `<div class='message user'>${userMessage} <span class="timestamp">${timestamp}</span></div>`
        );

        $(".typing").show(); // Show typing indicator

        $.ajax({
            url: "/get",
            type: "POST",
            data: { msg: userMessage },
            success: function (response) {
                $(".typing").hide(); // Hide typing indicator

                let botMessage = response.response || "Sorry, I couldn't find an answer.";
                let timestamp = new Date().toLocaleTimeString();

                $("#chat-box").append(
                    `<div class='message bot'>
                        <img src="https://cdn-icons-png.flaticon.com/512/3135/3135715.png" alt="Doctor">
                        ${botMessage} 
                        <span class="timestamp">${timestamp}</span>
                    </div>`
                );

                $("#chat-box").scrollTop($("#chat-box")[0].scrollHeight);
            }
        });

        $("#user-input").val("");
    }
});
