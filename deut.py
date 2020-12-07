#!/usr/bin/env pypy3

# Deuterium Discord bot created by portasynthinca3
# https://github.com/portasynthinca3/deuterium
# See README.md, section "using" for license details

# constant messages
SUPPORTING = '''
You can tell your friends about this bot, as well as:
Vote for it on DBL: https://top.gg/bot/733605243396554813/vote
Donate money on Patreon: https://patreon.com/portasynthinca3
'''

BOT_INFO = '''
Hi!
This project was created by portasynthinca3 (https://github.com/portasynthinca3).
The sort of "backbone" of it is the markovify (https://github.com/jsvine/markovify) library.
You can join our support server if you're experiencing any issues: https://discord.gg/N52uWgD
'''

NOTICE = '''
I didn't expect this surge of users when I created this bot. As such, I can't keep up with the amount of users I have anymore, this project requires a more powerful server. Please consider supporting us (**`/d support`**).
'''

UNKNOWN_CMD      = ':x: **Unknown command - type `/d help` for help**'
GEN_FAILURE      = ':x: **This model has been trained on too little messages to generate sensible ones. Check back again later or use `/d train` - see `/d help` for help**'
INVALID_ARG      = ':x: **(One of) the argument(s) is(/are) invalid - see `/d help` for help**'
ONE_ARG          = ':x: **This command requires one argument - see `/d help` for help**'
LEQ_ONE_ARG      = ':x: **This command requires zero or one arguments - see `/d help` for help**'
SETS_UPD         = '**Settings updated successfully** :white_check_mark:'
INVALID_CHANNEL  = ':x: **The channel is either invalid or not a channel at all - see `/d help` for help**'
INVALID_MSG_CNT  = ':x: **You must request no less than 1 or more than 1000 messages - see `/d help` for help**'
RETRIEVING_MSGS  = '**Training. It will take some time, be patient** :white_check_mark:'
JSON_DEC_FAILURE = ':x: **The model failed to load, probably because it became corrupt in a recent change to how the bot stores channel info. Please either do a `/d reset` or join our support server (https://discord.gg/N52uWgD)**'
TOO_MANY_MSGS    = ':x: **Too many messages requested (10 max)**'
ADMINISTRATIVE_C = ':x: **This command is administrative. Only people with the "administrator" privilege can execute it.**'
RESET_SUCCESS    = '**Successfully reset training data for this channel** :white_check_mark:'
AUTOGEN_DISABLED = '**Auto-generation disabled** :white_check_mark:'
AUTOGEN_SETTO    = '**Auto-generation rate set to %s** :white_check_mark:'

BATCH_SIZE = 100

# try importing libraries
import sys, os, threading, re, atexit
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
    from discord.ext.commands import Bot
except ImportError:
    print('The discord library is not installed. Install it by running:\npip install discord')
    exit()

# prepare constant embeds
EMBED_COLOR = 0xe6f916

HELP_CMDS_REGULAR = {
    'help':                 'sends this message',
    'status':               'tells you the current settings and stats',
    'train <count>':        'trains the model using the last <count> messages in this channel',
    'gen':                  'generates a message immediately',
    'gen <count>':          'generates <count> messages immediately',
    'gen #channel-mention': 'generates a message using the mentioned channel model',
    'ggen':                 'generates a message immediately (using the global model)',
    'support':              'ways to support this project',
    'privacy':              'our privacy policy',
    'uwu <enable/disable>': 'enable/disable the UwU mode (don\'t.)',
    'info':                 'who created this?'
}
HELP_CMDS_ADMIN = {
    'collect <yes/no>':    'allows or denies training using new messages',
    'gcollect <yes/no>':   'allows or denies training the global model using new messages',
    'reset':               'resets the training model',
    'autorate <rate>':     'sets the automatic generation rate (one bot message per each <rate> normal ones)',
    #'arole @role-mention': 'grants administrative permissions to the mentioned role. Only one role can have them at a time'
}

HELP = discord.Embed(title='Deuterium commands', color=EMBED_COLOR)
HELP.add_field(inline=False, name='*All comands start with `/d `*', value='*ex.: `/d help`*')

HELP.add_field(inline=False, name='REGULAR COMMANDS', value='Can be executed by anybody')
for c in HELP_CMDS_REGULAR:
    HELP.add_field(name=f'`{c}`', value=HELP_CMDS_REGULAR[c])

HELP.add_field(inline=False, name='ADMINISTRATIVE COMMANDS', value='Can be executed by server administrators')
for c in HELP_CMDS_ADMIN:
    HELP.add_field(name=f'`{c}`', value=HELP_CMDS_ADMIN[c])

HELP.add_field(inline=False, name='Notice', value=NOTICE)


PRIVACY = discord.Embed(title='Deuterium Privacy Policy', color=EMBED_COLOR)
PRIVACY.add_field(inline=False, name='1. SCOPE', value='''
This message describes the relationship between the Deuterium Discord bot ("Deuterium", "the bot", "bot"), its creator ("I", "me") and its Users ("you").''')
PRIVACY.add_field(inline=False, name='2. AUTHORIZATION', value='''
When you authorize the bot, it is added as a member of the server you've chosen. It has no access to your profile, direct messages or anything that is not related to the selected server.''')
PRIVACY.add_field(inline=False, name='3. DATA PROCESSING', value='''
Deuterium processes every message it receives, both in authorized server channels and in direct messages. I should note, however, that before your message goes directly to the code I wrote, it's first received by the discord.py library, which I trust due to it being open source.
When my code receives a direct message, it sends a simple response back and stops further processing.
Else, if the message is received in a server channel:
- if the message starts with `/d `, the bot treats it as a command and doesn't store it
- else, if this channel has its "collect" setting set to "yes", it trains the model on this message and saves the said model do disk
- if this channel has its "global collect" setting set to "yes", it trains the global model on this message and saves the said model do disk''')
PRIVACY.add_field(inline=False, name='4. DATA STORAGE', value='''
Deuterium stores the following data:
- Channel settings and statistics (is message collection allowed, the total number of collected messages, etc.). This data can be viewed using the `/d status` command
- The local Markov chain model, which consists of a set of probabilities of a word coming after another word
- The global Markov chain model, which stores content described above
Deuterium does **not** store the following data:
- Raw messages
- User IDs/nicknames/tags
- Any other data not mentioned in the `Deuterium stores the following data` list above''')
PRIVACY.add_field(inline=False, name='5. CONTACTING', value='''
You can contact me regarding any issues through E-Mail (`portasynthinca3@gmail.com`)''')
PRIVACY.add_field(inline=False, name='6. DATA REMOVAL', value='''
Due to the nature of Markov chains, it's unfortunately not possible to remove a certain section of the data I store. Only the whole model can be reset.
If you wish to reset the local model, you may use the `/d reset` command.
If you wish to reset the global model, please contact me (see section `5. CONTACTING`) and provide explanation and proof of why you think that should happen.''')
PRIVACY.add_field(inline=False, name='7. DATA DISCLOSURE', value='''
I do not disclose collected data to any third parties. Furthermore, I do not look at it myself. The access to my server is properly secured, therefore it's unlikely that a potential hacker could gain access to the data.''')





# check if there is a token in the environment variable list
if 'DEUT_TOKEN' not in os.environ:
    print('No bot token (BEUT_TOKEN) in the list of environment variables')
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
def get_msgs(channel, before, limit):
    global BATCH_SIZE, TOKEN
    # split the limit into batches
    if limit > BATCH_SIZE:
        msgs = []
        for i in range(math.ceil(limit / BATCH_SIZE)):
            result = get_msgs(channel, before, min(BATCH_SIZE, limit - (i * BATCH_SIZE)))
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
        return get_msgs(channel, before, limit)

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
    if id == 0:
        return

    save_channel(id)
    channels.pop(id, {})

    print('X-X', id) # unloaded

    chanel_timers.pop(id, None)

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

    # abort any unloading timers
    timer = chanel_timers.pop(id, None)
    if timer is not None:
        timer.clear()

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

    print(' G ', id)

    return generated_msg

# generates a message and sends it in a separate thread
async def generate_channel_threaded(chan, cnt=1, id=-1):
    if id == -1:
        id = chan.id

    try:
        final_msg = ''
        for i in range(cnt):
            final_msg += await generate_channel(id, chan.id) + '\n'

        await chan.send(final_msg)
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

        print(' + ', id)
            
    except:
        pass

# trains a channel on previous messages (takes some time, meant to be run in a separate thread)
def train_on_prev(chan_id, limit):
    msgs = get_msgs(chan_id, 0, limit)
    for m in msgs:
        train(chan_id, m['content'])





bot = Bot(command_prefix='/d ', help_command=None)

def on_bot_exit():
    # Thank you
    # I'll say goodbye soon
    # Though it's the end of the world
    # Don't blame yourself, no
    print('<-> CLOSING')
    for chan_id in channels:
        save_channel(chan_id)

@bot.event
async def on_ready():
    await load_channel(0, None)

    await bot.change_presence(activity=discord.Activity(
        type=discord.ActivityType.playing,
        name=f'with markov chains | "{bot.command_prefix}help" for help'
    ))
    print('Everything OK!')

    atexit.register(on_bot_exit)



@bot.command(pass_context=True, name='help')
async def help_cmd(ctx):
    await ctx.send(embed=HELP)

@bot.command(pass_context=True, name='privacy')
async def privacy_cmd(ctx):
    await ctx.send(embed=PRIVACY)

@bot.command(pass_context=True, name='support')
async def privacy_cmd(ctx):
    await ctx.send(SUPPORTING)

@bot.command(pass_context=True, name='info')
async def info_cmd(ctx):
    await ctx.send(BOT_INFO)

@bot.command(pass_context=True, name='status')
async def status_cmd(ctx):
    chan_info = channels[ctx.message.channel.id]
    autorate = chan_info['autorate']
    embed = discord.Embed(title='Deuterium status', color=EMBED_COLOR)
    embed.add_field(name='Training on messages from this channel',                    value='yes' if chan_info['collect'] else 'no')
    embed.add_field(name='Total messages trained on',                                 value=str(chan_info['collected']))
    embed.add_field(name='Contributing to the global model',                          value='yes' if chan_info['gcollect'] else 'no')
    embed.add_field(name='Messages added to the global model by this channel',        value=str(chan_info['gcollected']))
    embed.add_field(name='Messages added to the global model by everyone everywhere', value=str(channels[0]['collected']))
    embed.add_field(name='Automatically sending messages after each',                 value=str('[disabled]' if autorate == 0 else autorate))
    embed.add_field(name='UwU mode',                                                  value='enabled' if chan_info['uwumode'] else 'disabled')
    embed.add_field(name='Notice',                                                    value='Please see `/d help`')
    await ctx.send(embed=embed)

@bot.command(pass_context=True, name='uwu')
async def uwu_cmd(ctx, *args):
    if len(args) != 1:
        await ctx.send(ONE_ARG)
        return
    uwu = args[0]
    if uwu not in ['enable', 'disable']:
        await ctx.send(INVALID_ARG)
    else:
        channels[ctx.message.channel.id]['uwumode'] = (True
            if uwu == 'enable' else False)
        await ctx.send(SETS_UPD)

@bot.command(pass_context=True, name='gen')
async def gen_cmd(ctx, *args):
    channel = ctx.message.channel
    if len(args) == 0:
        await generate_channel_threaded(channel)

    elif len(args) == 1:
        try:
            count = int(args[0])
            if count > 10:
                await ctx.send(TOO_MANY_MSGS)
                return
            await generate_channel_threaded(channel, count)
        except: # it must be a channel
            try:
                target_cid = int(args[0][2:-1])
                await generate_channel_threaded(channel, 1, target_cid)
            except:
                await channel.send(INVALID_CHANNEL)

    else:
        await channel.send(LEQ_ONE_ARG)

@bot.command(pass_context=True, name='ggen')
async def ggen_cmd(ctx):
    # 0 because we're using the global model
    await generate_channel_threaded(ctx.message.channel, 1, 0)

@bot.command(pass_context=True, name='train')
async def train_cmd(ctx, *args):
    if len(args) != 1:
        await ctx.send(ONE_ARG)
        return
        
    try:
        cnt = int(args[0])
    except:
        await ctx.send(INVALID_ARG)
    else:
        if cnt > 1000 or cnt < 1:
            await ctx.send(INVALID_MSG_CNT)
            return
        await ctx.send(RETRIEVING_MSGS)
        retrieve_thr = threading.Thread(target=train_on_prev, args=(ctx.message.channel.id, cnt), name='Training thread')
        retrieve_thr.start()


# determines if the sender has admin privileges
def has_admin(ctx):
    author = ctx.message.author
    return author.guild_permissions.administrator

@bot.command(pass_context=True, name='collect')
async def collect_cmd(ctx, *args):
    if not has_admin(ctx):
        await ctx.send(ADMINISTRATIVE_C)
        return

    if len(args) != 1:
        await ctx.send(ONE_ARG)
        return

    collect = args[0]
    if collect not in ['yes', 'no']:
        await ctx.send(INVALID_ARG)
    else:
        channels[ctx.message.channel.id]['collect'] = True if collect == 'yes' else False
        await ctx.send(SETS_UPD)

@bot.command(pass_context=True, name='gcollect')
async def gcollect_cmd(ctx, *args):
    if not has_admin(ctx):
        await ctx.send(ADMINISTRATIVE_C)
        return
        
    if len(args) != 1:
        await ctx.send(ONE_ARG)
        return

    collect = args[0]
    if collect not in ['yes', 'no']:
        await ctx.send(INVALID_ARG)
    else:
        channels[ctx.message.channel.id]['gcollect'] = True if collect == 'yes' else False
        await ctx.send(SETS_UPD)

@bot.command(pass_context=True, name='reset')
async def reset_cmd(ctx):
    if not has_admin(ctx):
        await ctx.send(ADMINISTRATIVE_C)
        return

    chan_info = channels[ctx.message.channel.id]
    chan_info['model'] = None
    chan_info['collected'] = 0
    await ctx.send(RESET_SUCCESS)

@bot.command(pass_context=True, name='autorate')
async def autorate_cmd(ctx, *args):
    if not has_admin(ctx):
        await ctx.send(ADMINISTRATIVE_C)
        return
        
    if len(args) != 1:
        await ctx.send(ONE_ARG)
        return

    autorate = args[0]
    try:
        autorate = int(autorate)
        if autorate < 0:
            await ctx.send(INVALID_ARG)
            return
        chan_info['autorate'] = autorate
        await ctx.send(AUTOGEN_DISABLED if autorate == 0 else (AUTOGEN_SETTO % str(autorate)))
    except:
        await ctx.send(INVALID_ARG)

@bot.event
async def on_command_error(ctx, ex):
    if type(ex) is discord.ext.commands.errors.CommandNotFound:
        await ctx.send(UNKNOWN_CMD)



# we need this because the bot has to train its message generation models
@bot.event
async def on_message(msg: discord.Message):
    global channels
    channel = msg.channel

    # ignore bots
    if msg.author.bot:
        return

    # don't react to DMs
    if msg.guild == None:
        await channel.send(':x: **This bot only works in servers**')
        return

    # don't react to empty messages (if it's just a picture, or audio, etc.)
    if len(msg.content) == 0:
        return

    # load channel settings and model from the disk if available and needed
    chan_id = channel.id
    if channel_exists(chan_id) and chan_id not in channels:
        await load_channel(chan_id, channel)

    # create a new channel object if it doesn't exist
    if chan_id not in channels:
        chan_info = {'model':None,
                                'collect':True, 'collected':0,
                                'autorate':20, 'total_msgs':0,
                                'gcollect':False, 'gcollected':0,
                                'uwumode':False}
    
    chan_info = channels[chan_id]

    # add fields that appeared in newer versions of the bot
    if 'total_msgs' not in chan_info:
        chan_info['total_msgs'] = 0

    if 'uwumode'    not in chan_info:
        chan_info['uwumode'] = False

    # check if it's a command
    if msg.content.startswith(bot.command_prefix):
        # process the command and return
        await bot.process_commands(msg)
        return

    # it's an ordinary message and not a command
    # train on this message if allowed
    if chan_info['collect']:
        train(chan_id, msg.content)

    # train the global model if allowed
    if chan_info['gcollect']:
        train(0, msg.content)
        chan_info['gcollected'] += 1

    # generate a message if needed
    if chan_info['autorate'] > 0 and chan_info['total_msgs'] % chan_info['autorate'] == 0:
        await generate_channel_threaded(chan_id, chan_id, channel)

    chan_info['total_msgs'] += 1

    # unload the channel in a while
    schedule_unload(chan_id)

bot.run(TOKEN)