from twilio.rest import Client

# Your Account SID and Auth Token from console.twilio.com
account_sid = "AC06701fc775020d4e8534dcf44130cd73"
auth_token  = "040f7c16dc47542ea1989376bacccfdb"

client = Client(account_sid, auth_token)

message = client.messages.create(
    to="+6282227486067",
    from_="+12542806157",
    body="Hello from Python!")

print(message.sid)