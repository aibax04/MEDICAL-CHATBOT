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

        $("#chat-box").append("<p class='user'><b>You:</b> " + userMessage + "</p>");
        $("#user-input").val(""); // Clear input field

        $.post("/get", { msg: userMessage })
        .done(function(data) {
            console.log("🔄 Received Data:", data);  // ✅ Debugging log

            // ✅ Extract response from JSON object
            if (data && data.response) {
                $("#chat-box").append("<p class='bot'><b>Bot:</b> " + data.response + "</p>");
            } else {
                $("#chat-box").append("<p class='bot'><b>Bot:</b> ⚠ Error: No valid response received.</p>");
            }
        })
        .fail(function(xhr, status, error) {
            console.log("❌ AJAX Error:", error);
            $("#chat-box").append("<p class='bot'><b>Bot:</b> ❌ Error retrieving response.</p>");
        });
    }
});
