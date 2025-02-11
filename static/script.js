$(document).ready(function() {
    $("#send-button").click(function() {
        sendMessage();
    });

    $("#user-input").keypress(function(event) {
        if (event.which == 13) { // Enter key
            sendMessage();
        }
    });

    function sendMessage() {
        var userMessage = $("#user-input").val().trim();
        if (!userMessage) return;  // Ignore empty messages

        var timeStamp = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

        $("#chat-box").append("<div class='message user-message'><p><b>You:</b> " + userMessage + "</p><span class='time'>" + timeStamp + "</span></div>");
        $("#user-input").val(""); // Clear input field

        $.post("/get", { msg: userMessage })
        .done(function(data) {
            console.log("🔄 Response Data:", data);  

            if (data && data.response) {
                $("#chat-box").append("<div class='message bot-message'><p><b>Bot:</b> " + data.response + "</p><span class='time'>" + timeStamp + "</span></div>");
            } else {
                $("#chat-box").append("<div class='message bot-message'><p><b>Bot:</b> ⚠ No valid response.</p><span class='time'>" + timeStamp + "</span></div>");
            }
        })
        .fail(function(xhr, status, error) {
            console.log("❌ AJAX Error:", error);
            $("#chat-box").append("<div class='message bot-message'><p><b>Bot:</b> ❌ Error retrieving response.</p><span class='time'>" + timeStamp + "</span></div>");
        });

        $("#chat-box").scrollTop($("#chat-box")[0].scrollHeight); // Auto-scroll to the latest message
    }
});
