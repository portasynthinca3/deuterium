#!/usr/bin/env pypy3

# Deuterium Discord bot created by portasynthinca3
# https://github.com/portasynthinca3/deuterium
# See README.md, section "using" for license details

# constant messages
HELP = '''
**Deuterium commands**
`/d help` - send this message
`/d status` - tells you everything about this bot and what it knows about this channel
`/d train "limit"` - trains the model on the last "limit" messages in this channel
`/d collect yes/no` - allow/deny collecting messages for this channel
`/d gcollect yes/no` - allow/deny collecting messages from this this channel to contribute to the global model
`/d reset` - reset training data for this channel
`/d autorate "rate"` - sets the rate at which the bot would automatically generate messages (one after each "rate") (set to 0 do disable)
`/d gen` - immediately generate a message
`/d gen #channel-mention` - immediately generate a message using the model of the mentioned channel
`/d ggen` - immediately generate a message using the global model
`/d donate` - send donation information
`/d privacy` - send the privacy policy
`/d uwu enable/disable` - enables/disables the UwU mode
**Notice**
I didn't expect this surge of users when I created this bot. As such, I can't keep up with the amount of users I have anymore, this project requires a more powerful server. Please consider donating (**`/d donate`**).
'''

DONATING = '''
We appreciate your will to support us! It really helps us keep our servers running.
Patreon: https://patreon.com/portasynthinca3
'''

GEN_FAILURE      = ':x: **The model for this channel has been trained on too little messages to generate sensible ones. Check back again later or use `/d train` - see `/d help` for help**'
INVALID_ARG      = ':x: **(One of) the argument(s) is(/are) invalid - see `/d help` for help**'
ONE_ARG          = ':x: **This command requires one argument - see `/d help` for help**'
LEQ_ONE_ARG      = ':x: **This command requires zero or one arguments - see `/d help` for help**'
SETS_UPD         = '**Settings updated successfully** :white_check_mark:'
INVALID_CHANNEL  = ':x: **The channel is either invalid or not a channel at all - see `/d help` for help**'
INVALID_MSG_CNT  = ':x: **You must request no less than 1 or more than 1000 messages - see `/d help` for help**'
RETRIEVING_MSGS  = '**Training. It will take some time, be patient** :white_check_mark:'
JSON_DEC_FAILURE = ':x: **The model failed to load from disk, proably because it became corrupt in a recent change to how the bot stores channel info. Please either do a `/d reset` or contact the bot creator (`duck🦆#1746`)**'

BATCH_SIZE = 100

# try importing libraries
import sys, os, threading, re
import requests, json, gzip
import math, random
import time
import asyncio
from threading import Thread
from multiprocessing import Process
from discord.errors import Forbidden
from json import JSONDecodeError

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
chanel_timers = {}
channel_limits = {}

# retrieves messages from a channel
def getMsgs(channel, before, limit):
    global BATCH_SIZE, TOKEN
    # split the limit into batches
    if limit > BATCH_SIZE:
        msgs = []
        for i in range(math.ceil(limit / BATCH_SIZE)):
            result = getMsgs(channel, before, min(BATCH_SIZE, limit - (i * BATCH_SIZE)))
            msgs += result
            before = result[0]['id']
            time.sleep(0.25)
        return msgs
    
    # make the request
    beforeStr = '' if before == 0 else f'&before={before}'
    response = requests.get(f'https://discord.com/api/v8/channels/{channel}/messages?limit={limit}{beforeStr}',
        headers={ 'Authorization': 'Bot ' + TOKEN }).json()

    if isinstance(response, dict):
        # rate limiting
        print(response)
        time.sleep(response['retry_after'] + 0.1)
        return getMsgs(channel, before, limit)

    # return simplified data
    return [{'id':int(x['id']), 'content':x['content']} for x in response]

# sets a unload timer
def schedule_unload(id):
    if id == 0:
        return

    timer = chanel_timers.pop(id, None)

    if timer is not None:
        timer.cancel()

    timer = threading.Timer(30, unload_channel, (id,))
    timer.start()
    chanel_timers[id] = timer

# unloads a channel from memory
def unload_channel(id):
    save_channel(id)
    channels.pop(id, {})

    print('X-X', id) # unloaded

# saves channel settings and model
def save_channel(id):
    global channels
    with gzip.open(f'channels/{str(id)}.json.gz', 'wb') as f:
        to_jsonify = channels[id].copy()
        if to_jsonify['model'] != None:
            to_jsonify['model'] = to_jsonify['model'].to_dict()
        f.write(json.dumps(to_jsonify).encode('utf-8'))

    print('<--', id) # dumped

def channel_exists(id):
    return os.path.exists(f'channels/{str(id)}.json.gz')

# loads channel settings and model
async def load_channel(id, channel):
    global channels
    with gzip.open(f'channels/{str(id)}.json.gz', 'rb') as f:
        try:
            jsonified = json.loads(f.read().decode('utf-8'))

            if jsonified['model'] != None:
                jsonified['model'] = markovify.NewlineText.from_dict(jsonified['model'])
            channels[id] = jsonified

        except JSONDecodeError:
            print('!!!', id) # failed to load
            if channel is not None:
                await channel.send(JSON_DEC_FAILURE)
                return

    print('-->', id) # loaded

# generates a message
async def generate_channel(id, act_id):
    global channels

    if channel_exists(id) and id not in channels:
        await load_channel(id, None)

    if channels[id]['model'] == None:
        return GEN_FAILURE

    generated_msg = channels[id]['model'].make_short_sentence(280, tries=50)
    if generated_msg == None or len(generated_msg.replace('\n', ' ').strip()) == 0:
        return GEN_FAILURE

    # apply the (optional) UwU filter
    if channels[act_id]['uwumode']:
        generated_msg = generated_msg.replace('r', 'w')
        generated_msg = generated_msg.replace('R', 'W')

        generated_msg = generated_msg.replace('l', 'w')
        generated_msg = generated_msg.replace('L', 'W')

        generated_msg = generated_msg.replace('owo', 'OwO')
        generated_msg = generated_msg.replace('uwu', 'UwU')

        generated_msg += ' UwU~' if bool(random.getrandbits(1)) else ' OwO~'

    # remove mentions (incl. channels, etc.) from messages generated from the global model
    if id == 0:
        generated_msg = re.sub('<(@|&)!*[0-9]*>', '**[mention removed]**', generated_msg)

    return generated_msg

# generates a message and sends it in a separate thread
async def generate_channel_threaded(id, act_id, chan):
    try:
        await chan.send(await generate_channel(id, act_id))
    except Forbidden:
        pass

# trans a model on a message
def train(id, text):
    try:
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
    except:
        pass

# trains a channel on previous messages (takes some time, meant to be run in a separate thread)
def train_on_prev(chan_id, limit):
    msgs = getMsgs(chan_id, 0, limit)
    for m in msgs:
        train(chan_id, m['content'])

# the main client class
class Deuterium(discord.Client):

    async def on_ready(self):
        global SELF_ID
        SELF_ID = self.user.id

        await load_channel(0, None)

        await self.change_presence(activity=discord.Game(name='with markov chains'))
        print('Everything OK!')

    async def on_message(self, msg):
        global channels, SELF_ID
        # don't react to own messages
        if msg.author.id == SELF_ID:
            return

        # ignore bots
        if msg.author.bot:
            return

        # don't react to DMs
        if msg.guild == None:
            await msg.channel.send(':x: **This bot only works in servers**')
            return

        # don't react to empty messages (if it's just a picture, or audio, etc.)
        if len(msg.content) == 0:
            return

        # load channel settings and model from the disk if available and needed
        chan_id = msg.channel.id
        if channel_exists(chan_id) and chan_id not in channels:
            await load_channel(chan_id, msg.channel)

        # create a new channel object if it doesn't exist
        if chan_id not in channels:
            channels[chan_id] = {'model':None,
                                 'collect':True, 'collected':0,
                                 'autorate':20, 'total_msgs':0,
                                 'gcollect':False, 'gcollected':0,
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
                embed = discord.Embed(title='Deuterium status', color=0xe6f916)
                embed.add_field(inline=True, name='Training on messages from this channel',                          value='yes' if channels[chan_id]['collect'] else 'no')
                embed.add_field(inline=True, name='Total messages trained on',                                       value=str(channels[chan_id]['collected']))
                embed.add_field(inline=True, name='Contributing to the global model',                                value='yes' if channels[chan_id]['gcollect'] else 'no')
                embed.add_field(inline=True, name='Messages contributed to the global model by this channel',        value=str(channels[chan_id]['gcollected']))
                embed.add_field(inline=True, name='Messages contributed to the global model by everyone everywhere', value=str(channels[0]['collected']))
                embed.add_field(inline=True, name='Automatically sending messages after each',                       value=str('[disabled]' if autorate == 0 else autorate))
                embed.add_field(inline=True, name='UwU mode',                                                        value='enabled' if channels[chan_id]['uwumode'] else 'disabled')
                await msg.channel.send(embed=embed)

            elif cmd == 'collect':
                if len(args) != 1:
                    await msg.channel.send()
                else:
                    collect = args[0]
                    if collect not in ['yes', 'no']:
                        await msg.channel.send(INVALID_ARG)
                    else:
                        channels[chan_id]['collect'] = True if collect == 'yes' else False
                        await msg.channel.send(SETS_UPD)

            elif cmd == 'gcollect':
                if len(args) != 1:
                    await msg.channel.send(ONE_ARG)
                else:
                    collect = args[0]
                    if collect not in ['yes', 'no']:
                        await msg.channel.send(INVALID_ARG)
                    else:
                        channels[chan_id]['gcollect'] = True if collect == 'yes' else False
                        await msg.channel.send(SETS_UPD)

            elif cmd == 'uwu':
                if len(args) != 1:
                    await msg.channel.send(ONE_ARG)
                else:
                    uwu = args[0]
                    if uwu not in ['enable', 'disable']:
                        await msg.channel.send(INVALID_ARG)
                    else:
                        channels[chan_id]['uwumode'] = True if uwu == 'enable' else False
                        await msg.channel.send(SETS_UPD)

            elif cmd == 'gen':
                if len(args) == 0:
                    await generate_channel_threaded(chan_id, chan_id, msg.channel)
                elif len(args) == 1:
                    try:
                        target_cid = int(args[0][2:-1])
                        await generate_channel_threaded(target_cid, chan_id, msg.channel)
                    except:
                        await msg.channel.send(INVALID_CHANNEL)
                else:
                    await msg.channel.send(LEQ_ONE_ARG)

            elif cmd == 'ggen':
                await generate_channel_threaded(0, chan_id, msg.channel)

            elif cmd == 'privacy':
                embed = discord.Embed(title='Deuterium Privacy Policy', color=0xe6f916)
                embed.add_field(inline=False, name='1. SCOPE', value='''
This message describes the relationship between the Deuterium Discord bot ("Deuterium", "the bot", "bot"), its creator ("I", "me") and its Users ("you").''')
                embed.add_field(inline=False, name='2. AUTHORIZATION', value='''
When you authorize the bot, it is added as a member of the server you've chosen. It has no access to your profile, direct messages or anything that is not related to the selected server.''')
                embed.add_field(inline=False, name='3. DATA PROCESSING', value='''
Deuterium processes every message it receives, both in authorized server channels and in direct messages. I should note, however, that before your message goes directly to the code I wrote, it's first received by the discord.py library, which I trust due to it being open source.
When my code receives a direct message, it sends a simple response back and stops further processing.
Else, if the message is received in a server channel:
- if the message starts with `/d `, the bot treats it as a command and doesn't store it
- else, if this channel has its "collect" setting set to "yes", it trains the model on this message and saves the said model do disk
- if this channel has its "global collect" setting set to "yes", it trains the global model on this message and saves the said model do disk''')
                embed.add_field(inline=False, name='4. DATA STORAGE', value='''
Deuterium stores the following data:
- Channel settings and statistics (is message collection allowed, the total number of collected messages, etc.). This data can be viewed using the `/d status` command
- The local Markov chain model, which consists of a set of probabilities of a word coming after another word
- The global Markov chain model, which stores content described above
Deuterium does **not** store the following data:
- Raw messages
- User IDs/nicknames/tags
- Any other data not mentioned in the `Deuterium stores the following data` list above''')
                embed.add_field(inline=False, name='5. CONTACTING', value='''
You can contact me regarding any issues through E-Mail (`portasynthinca3@gmail.com`)''')
                embed.add_field(inline=False, name='6. DATA REMOVAL', value='''
Due to the nature of Markov chains, it's unfortunately not possible to remove a certain section of the data I store. Only the whole model can be reset.
If you wish to reset the local model, you may use the `/d reset` command.
If you wish to reset the global model, please contact me (see section `5. CONTACTING`) and provide explanation and proof of why you think that should happen.''')
                embed.add_field(inline=False, name='7. DATA DISCLOSURE', value='''
I do not disclose collected data to any third parties. Furthermore, I do not look at it myself. The access to my server is properly secured, therefore it's unlikely that a potential hacker could gain access to the data.''')
                await msg.channel.send(embed=embed)

            elif cmd == 'reset':
                channels[chan_id]['model'] = None
                channels[chan_id]['collected'] = 0
                await msg.channel.send('**Successfully reset training data for this channel** :white_check_mark:')

            elif cmd == 'donate':
                await msg.channel.send(DONATING)

            elif cmd == 'train':
                if len(args) != 1:
                    await msg.channel.send(ONE_ARG)
                else:
                    try:
                        cnt = int(args[0])
                    except:
                        await msg.channel.send(INVALID_ARG)
                    else:
                        if cnt > 1000 or cnt < 1:
                            await msg.channel.send(INVALID_MSG_CNT)
                        else:
                            await msg.channel.send(RETRIEVING_MSGS)
                            retrieve_thr = threading.Thread(target=train_on_prev, args=(chan_id, cnt), name='Previous training thread')
                            retrieve_thr.start()

            elif cmd == 'autorate':
                if len(args) != 1:
                    await msg.channel.send(ONE_ARG)
                else:
                    autorate = args[0]
                    try:
                        autorate = int(autorate)
                    except:
                        await msg.channel.send(INVALID_ARG)
                    else:
                        if autorate < 0:
                            await msg.channel.send(INVALID_ARG)
                        else:
                            channels[chan_id]['autorate'] = autorate
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
            await generate_channel_threaded(chan_id, chan_id, msg.channel)

        channels[chan_id]['total_msgs'] += 1

        schedule_unload(chan_id)

# create the client
deut = Deuterium()
deut.run(TOKEN)
