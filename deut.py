#!/usr/bin/env python3

# Deuterium Discord bot

# constant messages
HELP = '''
**Deuterium commands**
`/d help` - send this message
`/d status` - print the information about this channel and the global model
`/d collect yes/no` - allow/disallow collecting messages for this channel
`/d gcollect yes/no` - allow/disallow collecting messages from this this channel to contribute to the global model
`/d gen` - immediately generate a message
`/d ggen` - immediately generate a message using the global model
`/d privacy` - send the privacy policy
`/d reset` - reset training data for this channel
`/d autorate rate` - sets the rate at which the bot would automatically generate messages (set to 0 do disable)
`/d uwu enable/disable` - enables/disables UwU mode
'''

PRIVACY = '''
**Deuterium privacy policy**
**1. SCOPE**
This message describes the relationship between the Deuterium Discord bot ("Deuterium", "the bot", "bot"), its creator ("I", "me") and its Users ("you").
**2. AUTHORIZATION**
When you authorize the bot, it is added as a member of the server you've chosen. The bot has no access to your profile, direct messages or anything that is not related to the selected server.
**3. DATA PROCESSING**
Deuterium processes every message it receives, both in authorized server channels and in direct messages. When it receives a direct message, it sends a simple response back and stops further processing.
Else, if the message is received in a server channel:
- if the message starts with `/d `, the bot treats it as a command and doesn't store it
- else, if this channel has its "collect" setting set to "yes", it trains the internal model on this message and saves the said model do disk
**4. DATA STORAGE**
Deuterium stores the following data:
- Channel settings and statistics (is message collection allowed, the total number of collected messages)
- Markov chain model, which includes the connection between words and their probabilities
Deuterium does _not_ store the following data:
- Raw messages
- User IDs/nicknames/tags
- Any other data not mentioned in the `Deuterium stores the following data` list above
**5. CONTACTING**
You can contact me through Discord (`portasynthinca3#1746`), E-Mail (`portasynthinca3@gmail.com`), or GitHub (`@portasynthinca3`)
'''

GEN_FAILURE = ':x: **The model for this channel had been trained on too little messages to generate sensible messages. Try speaking for longer and check back again later**'

# try importing libraries
import os, json, random

try:
    import markovify
except ImportError:
    print('The markovify library is not installed. Install it by running:\npip install markovify')
    exit()

try:
    import discord
except ImportError:
    print('The discord library is not installed. Install it by running:\npip install discord')
    exit()

# check if there is a token in the environment variable list
if 'DEUT_TOKEN' not in os.environ:
    print('No bot token in the list of environment variables')
    exit()
TOKEN = os.environ['DEUT_TOKEN']

# create a channel list folder
if not os.path.exists('channels'):
    os.mkdir('channels')

# some global variables
SELF_ID = None
channels = {}

# saves channel settings and model
def save_channel(id):
    global channels
    with open(f'channels/{str(id)}.json', 'w') as f:
        to_jsonify = channels[id].copy()
        if to_jsonify['model'] != None:
            to_jsonify['model'] = to_jsonify['model'].to_dict()
        json.dump(to_jsonify, f)

# loads channel settings and model
def load_channel(id):
    global channels
    with open(f'channels/{str(id)}.json', 'r') as f:
        jsonified = json.load(f)
        if jsonified['model'] != None:
            jsonified['model'] = markovify.NewlineText.from_dict(jsonified['model'])
        channels[id] = jsonified

# generates a message
def generate_channel(id, act_id):
    if channels[id]['model'] == None:
        return GEN_FAILURE

    generated_msg = channels[id]['model'].make_short_sentence(280, tries=50)
    if generated_msg == None or len(generated_msg.replace('\n', ' ').strip()) == 0:
        return GEN_FAILURE

    # apply the (optional) UwU filter
    if channels[act_id]['uwumode']:
        generated_msg = generated_msg.replace('r', 'w')
        generated_msg = generated_msg.replace('R', 'W')
        generated_msg = generated_msg.replace('l', 'ww')
        generated_msg = generated_msg.replace('L', 'WW')
        generated_msg += ' UwU~' if bool(random.getrandbits(1)) else ' OwO~'

    return generated_msg


# trans a model on a message
def train(id, text):
    # join period-separated sentences by a new line
    text = '\n'.join(text.split('.'))

    # create a new model if it doesn't exist
    if channels[id]['model'] == None:
        channels[id]['model'] = markovify.NewlineText(text)

    # create a model with this message and combine it with the already existing one ("train" the model)
    new_model = markovify.NewlineText(text)
    channels[id]['model'] = markovify.combine([channels[id]['model'], new_model])

    # increment the number of collected messages
    channels[id]['collected'] += 1

    # save the model to disk
    save_channel(id)

# the main client class
class Deuterium(discord.Client):

    async def on_ready(self):
        global SELF_ID
        SELF_ID = self.user.id

        load_channel(0)

        await self.change_presence(activity=discord.Game(name='markov chains'))
        print('Everything OK!')

    async def on_message(self, msg):
        global channels, SELF_ID
        # don't react to own messages
        if msg.author.id == SELF_ID:
            return

        # don't react to messages in DMs
        if msg.guild == None:
            await msg.channel.send(':x: **This bot only works in servers**')
            return

        # don't react to empty messages (if it's just a picture, or audio, etc.)
        if len(msg.content) == 0:
            return

        # load channel settings and model from the disk if available and needed
        chan_id = msg.channel.id
        if os.path.exists(f'channels/{str(chan_id)}.json') and chan_id not in channels:
            load_channel(chan_id)

        # create a new channel object if it doesn't exist
        if chan_id not in channels:
            channels[chan_id] = {'model':None,
                                 'collect':True, 'collected':0,
                                 'autorate':20, 'total_msgs':0,
                                 'gcollect':True, 'gcollected':0,
                                 'uwumode':False}

        # add fields that appeared in newer versions of the bot
        if 'total_msgs' not in channels[chan_id]:
            channels[chan_id]['total_msgs'] = 0
        if 'uwumode' not in channels[chan_id]:
            channels[chan_id]['uwumode'] = False

        # check if it's a command
        if msg.content.startswith('/d '):
            # get the command and arguments
            args = msg.content[len('/d '):].split(' ')
            cmd = args[0]
            args = args[1:]

            if cmd == 'help':
                await msg.channel.send(HELP)

            elif cmd == 'status':
                autorate = channels[chan_id]['autorate']
                status_msg = ('Collecting messages from this channel: **' + ('yes' if channels[chan_id]['collect'] else 'no') + '**\n'
                              'Total collected messages: **' + str(channels[chan_id]['collected']) + '**\n'
                              'Contributing to the global model: **' + ('yes' if channels[chan_id]['gcollect'] else 'no') + '**\n'
                              'Messages contributed to the global model: **' + str(channels[chan_id]['gcollected']) + '**\n'
                              'Messages contributed to the global model by everyone: **' + str(channels[0]['collected']) + '**\n'
                              'Automatically sending messages after: **' + str('disabled' if autorate == 0 else autorate) + '**\n'
                              'UwU mode: **' + ('enabled' if channels[chan_id]['uwumode'] else 'disabled') + '**\n')
                await msg.channel.send(status_msg)

            elif cmd == 'collect':
                if len(args) != 1:
                    await msg.channel.send(':x: **This command requires one argument - type `/d help` for help**')
                else:
                    collect = args[0]
                    if collect not in ['yes', 'no']:
                        await msg.channel.send(':x: **(One of) the argument(s) is(/are) invalid - type `/d help` for help**')
                    else:
                        channels[chan_id]['collect'] = True if collect == 'yes' else False
                        await msg.channel.send('**Successfully set permissions for this channel** :white_check_mark:')
                        save_channel(chan_id)

            elif cmd == 'gcollect':
                if len(args) != 1:
                    await msg.channel.send(':x: **This command requires one argument - type `/d help` for help**')
                else:
                    collect = args[0]
                    if collect not in ['yes', 'no']:
                        await msg.channel.send(':x: **(One of) the argument(s) is(/are) invalid - type `/d help` for help**')
                    else:
                        channels[chan_id]['gcollect'] = True if collect == 'yes' else False
                        await msg.channel.send('**Successfully set permissions for this channel** :white_check_mark:')
                        save_channel(chan_id)

            elif cmd == 'uwu':
                if len(args) != 1:
                    await msg.channel.send(':x: **This command requires one argument - type `/d help` for help**')
                else:
                    uwu = args[0]
                    if uwu not in ['enable', 'disable']:
                        await msg.channel.send(':x: **(One of) the argument(s) is(/are) invalid - type `/d help` for help**')
                    else:
                        channels[chan_id]['uwumode'] = True if uwu == 'enable' else False
                        await msg.channel.send('**Successfully updated settings for this channel** :white_check_mark:')
                        save_channel(chan_id)

            elif cmd == 'gen':
                await msg.channel.send(generate_channel(chan_id, chan_id))

            elif cmd == 'ggen':
                await msg.channel.send(generate_channel(0, chan_id))

            elif cmd == 'privacy':
                await msg.channel.send(PRIVACY)

            elif cmd == 'reset':
                channels[chan_id]['model'] = None
                channels[chan_id]['collected'] = 0
                await msg.channel.send('**Successfully reset training data for this channel** :white_check_mark:')

            elif cmd == 'autorate':
                if len(args) != 1:
                    await msg.channel.send(':x: **This command requires one argument - type `/d help` for help**')
                else:
                    autorate = args[0]
                    try:
                        autorate = int(autorate)
                    except:
                        await msg.channel.send(':x: **(One of) the argument(s) is(/are) invalid - type `/d help` for help**')
                    else:
                        if autorate < 0:
                            await msg.channel.send(':x: **(One of) the argument(s) is(/are) invalid - type `/d help` for help**')
                        else:
                            channels[chan_id]['autorate'] = autorate
                            save_channel(chan_id)
                            if autorate == 0:
                                await msg.channel.send('**Auto-generation disabled** :white_check_mark:')
                            else:
                                await msg.channel.send(f'**Auto-generation rate set to {str(autorate)}** :white_check_mark:')

            else:
                await msg.channel.send(':x: **Unknown command - type `/d help` for help**')

            return # don't train on commands

        # it's an ordinary message and not a command
        # train on this message if allowed
        if channels[chan_id]['collect']:
            train(chan_id, msg.content)

        # train the global model if allowed
        if channels[chan_id]['gcollect']:
            train(0, msg.content)
            channels[chan_id]['gcollected'] += 1

        # generate a message if needed
        if channels[chan_id]['autorate'] > 0 and channels[chan_id]['total_msgs'] % channels[chan_id]['autorate'] == 0:
            await msg.channel.send(generate_channel(chan_id, chan_id))

        channels[chan_id]['total_msgs'] += 1
            
        save_channel(chan_id)

# create the client
deut = Deuterium()
deut.run(TOKEN)