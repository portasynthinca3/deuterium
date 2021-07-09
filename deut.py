#!/usr/bin/env pypy3

# Deuterium Discord bot created by portasynthinca3
# https://github.com/portasynthinca3/deuterium
# See README.md, section "using" for license details

# constant messages
SUPPORTING = '''
You can tell your friends about this bot, as well as:
Vote on DBL: https://top.gg/bot/733605243396554813/vote
Donate money on Patreon: https://patreon.com/portasynthinca3
'''

BOT_INFO = '''
Hi!
This project was created by portasynthinca3 (https://github.com/portasynthinca3).
The sort of "backbone" of it is the markovify library (https://github.com/jsvine/markovify).
You can join our support server if you're experiencing any issues: https://discord.gg/N52uWgD.
Please consider supporting us (`!!d support`).
'''

ANNOUNCEMENT = '''
The global model just got reset on Jan 27th!
'''

MOODS = [
    {
        'title':             ':cry: Depressed',
        'game_response_pos': [],
        'game_response_neg': ['I don\'t really want to play now'],
        'suggest_chance':    0,
        'pos_chance':        0,
        'treat_chance':      0,
        'treats':            []
    },
    {
        'title':             ':pensive: Sad',
        'game_response_pos': ['Uhm, okay...'],
        'game_response_neg': ['Not now, sorry'],
        'suggest_chance':    0,
        'pos_chance':        0.2,
        'treat_chance':      0,
        'treats':            []
    },
    {
        'title':             ':neutral_face: Neutral',
        'game_response_pos': ['Sure, but I have to go soon...'],
        'game_response_neg': ['Meh, I\'ve got some homework to do, sorry'],
        'suggest_chance':    0.0025,
        'pos_chance':        0.5,
        'treat_chance':      0.005,
        'treats':            ['candy :candy:', 'chocolate bar :chocolate_bar:']
    },
    {
        'title':             ':smile: Happy',
        'game_response_pos': ['Of course, let\'s go!'],
        'game_response_neg': ['I\'d love to have a game with you, but unfortunately I can\'t right now. Wanna play later?'],
        'suggest_chance':    0.005,
        'pos_chance':        0.95,
        'treat_chance':      0.01,
        'treats':            ['candy :candy:', 'chocolate bar :chocolate_bar:', 'lollipop :lollipop:']
    }
]

NEW_PREFIX       = ':information_source: **The command prefix has been changed to `!!d ` to avoid collision with Discord slash commands**'
UNKNOWN_CMD      = ':x: **Unknown command - type `!!d help` for help**'
GEN_FAILURE      = ':x: **This model has been trained on too little messages to generate sensible ones. Check back again later or use `!!d train` - see `!!d help` for help**'
INVALID_ARG      = ':x: **(One of) the argument(s) is(/are) invalid - see `!!d help` for help**'
ONE_ARG          = ':x: **This command requires one argument - see `!!d help` for help**'
LEQ_ONE_ARG      = ':x: **This command requires zero or one arguments - see `!!d help` for help**'
SETS_UPD         = '**Settings updated successfully** :white_check_mark:'
INVALID_CHANNEL  = ':x: **The channel is either invalid or not a channel at all - see `!!d help` for help**'
INVALID_MSG_CNT  = ':x: **You must request no less than 1 or more than 1000 messages - see `!!d help` for help**'
RETRIEVING_MSGS  = '**Training. It will take some time, be patient** :white_check_mark:'
JSON_DEC_FAILURE = ':x: **The model failed to load, probably because it became corrupted as a result of me carelessly shutting the bot down, a power outage or other things, I\'m sorry. Please either do a `!!d reset` or join our support server (<https://discord.gg/N52uWgD>)**'
TOO_MANY_MSGS    = ':x: **Too many messages requested (10 max)**'
ADMINISTRATIVE_C = ':x: **This command is administrative. Only people with the "administrator" privilege can execute it**'
RESET_SUCCESS    = '**Successfully reset training data for this channel** :white_check_mark:'
AUTOGEN_DISABLED = '**Auto-generation disabled** :white_check_mark:'
AUTOGEN_SETTO    = '**Auto-generation rate set to {0}** :white_check_mark:'
CREATOR_ONLY     = ':x: **This command can only be executed by the creator of the bot**'
RPS_ERROR        = ':x: **I\'m sorry, {0} is playing Rock-Paper-Scissors with me already**'
RPS_INSTRUCTIONS = '**Alright, say `rock`, `paper` or `scissors`!** :white_check_mark:'
RPS_BOTS_MOVE    = 'Alright, I go with **{0}**'
RPS_USER         = '**You won :clap:**'
RPS_DRAW         = '**It\'s a draw :pushpin:**'
RPS_BOT          = '**I won :smiley:**'
RPS_SUGGESTION   = '**Hey {0}! Do you want to play Rock-Paper-Scissors with me? If so, type `!!d rps`** :smiley:'

BATCH_SIZE = 100

# try importing libraries
import sys, os, threading, re, string, atexit, psutil, gc, traceback
import requests, json, gzip, zlib, heapq
import math, random
import time, datetime
import asyncio
import markovify
import discord
from discord.ext.commands import Bot
from threading import Thread
from multiprocessing import Process
from discord.errors import Forbidden
from json import JSONDecodeError

# prepare constant embeds
EMBED_COLOR = 0xe6f916

HELP_CMDS_REGULAR = {
    'help':                     ':information_source: sends this message',
    'status':                   ':green_circle: tells you the current settings and stats',
    'stats':                    ':yellow_circle: Deuterium resource usage',
    'gen':                      ':new: generates a message immediately',
    'gen <count>':              ':1234: generates <count> messages immediately',
    'gen #channel-mention':     ':level_slider: generates a message using the mentioned channel model',
    'ggen':                     ':rocket: generates a message immediately (using the global model)',
    'support':                  ':question: ways to support this project',
    'privacy':                  ':lock: our privacy policy',
    'uwu <enable/disable>':     ':eyes: enable/disable the UwU mode',
    'info':                     ':thinking: who created this?',
    'scoreboard':               ':100: top 10 most active users',
    'rps':                      ':rock: starts a game of Rock-Paper-Scissors with the bot'
}
HELP_CMDS_ADMIN = {
    'collect <yes/no>':         ':book: allows or denies learning new messages',
    'gcollect <yes/no>':        ':books: allows or denies learning new messages to contribute to the global model',
    'actions <yes/no>':         ':candy: enables or disables "actions" such as giving out candies',
    'reset':                    ':rotating_light: resets everything',
    'autorate <rate>':          ':bar_chart: sets the automatic generation rate (one bot message per each <rate> normal ones +/- half of this value)',
    'train <count>':            ':books: trains the model using the last <count> messages in this channel',
    'ignore_bots <yes/no>':     ':robot: whether or not to ignore other bots, servers and webhooks',
    'remove_mentions <yes/no>': ':mailbox: whether or not to remove mentions in generated messages. Those generated from the global model always have them removed'
}

# help embed
HELP = discord.Embed(title='Deuterium commands', color=EMBED_COLOR)
HELP.add_field(inline=False, name=':loudspeaker: Latest Announcement', value=ANNOUNCEMENT)

HELP.add_field(inline=False, name=':exclamation: *All comands start with `!!d `*', value='*ex.: `!!d help`*')

HELP.add_field(inline=False, name='REGULAR COMMANDS', value='Can be executed by anybody')
for c in HELP_CMDS_REGULAR:
    HELP.add_field(name=f'`{c}`', value=HELP_CMDS_REGULAR[c])

HELP.add_field(inline=False, name='ADMINISTRATIVE COMMANDS', value='Can be executed by server administrators')
for c in HELP_CMDS_ADMIN:
    HELP.add_field(name=f'`{c}`', value=HELP_CMDS_ADMIN[c])



PRIVACY = discord.Embed(title='Deuterium Privacy Policy', color=EMBED_COLOR)
PRIVACY.add_field(inline=False, name='1. SCOPE', value='''
This message describes the relationship between the Deuterium Discord bot ("Deuterium", "the bot", "bot"), its creator ("I", "me") and its Users ("you").''')
PRIVACY.add_field(inline=False, name='2. AUTHORIZATION', value='''
When you authorize the bot, it is added as a member of the server you've chosen. It has no access to your profile, direct messages or anything that is not related to the selected server.''')
PRIVACY.add_field(inline=False, name='3. DATA PROCESSING', value='''
Deuterium processes every message it receives, both in authorized server channels and in direct messages. I should note, however, that before your message goes directly to the code I wrote, it's first received by the discord.py library, which I trust due to it being open source.
When my code receives a direct message, it sends a simple response back and stops further processing.
Else, if the message is received in a server channel:
- if the message starts with `!!d `, the bot treats it as a command and doesn't store it
- else, if this channel has its "collect" setting set to "yes", it trains the model on this message and saves the said model do disk
- if this channel has its "global collect" setting set to "yes", it trains the global model on this message and saves the said model do disk''')
PRIVACY.add_field(inline=False, name='4. DATA STORAGE', value='''
Deuterium stores the following data:
- Channel settings and statistics (is message collection allowed, the total number of collected messages, etc.). This data can be viewed using the `!!d status` command
- The local Markov chain model, which consists of a set of probabilities of a word coming after another word
- The global Markov chain model, which stores content described above
- Channel and user IDs
Deuterium does **not** store the following data:
- Raw messages
- User nicknames/tags
- Any other data not mentioned in the `Deuterium stores the following data` list above''')
PRIVACY.add_field(inline=False, name='5. CONTACTING', value='''
You can contact me regarding any issues through E-Mail (`portasynthinca3@gmail.com`)''')
PRIVACY.add_field(inline=False, name='6. DATA REMOVAL', value='''
Due to the nature of Markov chains, it's unfortunately not possible to remove a certain section of the data I store. Only the whole model can be reset.
If you wish to reset the local model, you may use the `!!d reset` command.
If you wish to reset the global model, please contact me (see section `5. CONTACTING`) and provide explanation and proof of why you think that should happen.''')
PRIVACY.add_field(inline=False, name='7. DATA DISCLOSURE', value='''
I do not disclose collected data to any third parties. Furthermore, I do not look at it myself. The access to my server is properly secured, therefore it's unlikely that a potential hacker could gain access to the data.''')




# check if there is a token in the environment variable list
if 'DEUT_TOKEN' not in os.environ:
    print('No bot token (DEUT_TOKEN) in the list of environment variables')
    exit()
TOKEN = os.environ['DEUT_TOKEN']

# create a channel list folder
if not os.path.exists('channels'):
    os.mkdir('channels')

bot = Bot(command_prefix='!!d ', help_command=None)
slash_prefix = bot.command_prefix[1:]




# some global variables
SELF_ID = None
channels = {}
channel_timers = {}
channel_limits = {}
channel_locks = {}
process = None
global_c_lock = threading.Lock()

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
        headers={'Authorization': 'Bot ' + TOKEN}).json()

    if isinstance(response, dict):
        # rate limiting
        print(response)
        time.sleep(response['retry_after'] + 0.1)
        return get_msgs(channel, before, limit)

    # return simplified data
    return [{'id':int(x['id']), 'content':x['content']} for x in response]

# increases the mood value
def channel_add_mood(id, mood):
    chan_info = channels[id]
    chan_info['mood'] += mood
    chan_info['mood'] = min(100, max(0, chan_info['mood']))

# gets the channel mood
def channel_mood(id):
    return MOODS[min(3, channels[id]['mood'] * len(MOODS) // 100)]

# starts a R-P-S game
async def channel_game_start(channel, author):
    chan_info = channels[channel.id]

    # do we really want to play?
    mood = channel_mood(channel.id)
    if random.random() >= mood['pos_chance']:
        await channel.send(random.choice(mood['game_response_neg']))
        return
    
    # if not, set the author and send instructions
    chan_info['rps_id'] = author.id
    await channel.send('**' + random.choice(mood['game_response_pos']) + '**\n' + RPS_INSTRUCTIONS)

# processes a R-P-S game
async def channel_game_process(channel, msg, author):
    chan_info = channels[channel.id]

    # word-to-num map and score matrix
    rps_map    = {'rock': 0, 'paper': 1, 'scissors': 2}
    rps_revmap = {0: 'rock', 1: 'paper', 2: 'scissors'}
    rps_matrix = [[0, 1, -1], [-1, 0, 1], [1, -1, 0]]

    # check if the message _contains_ one of three terms
    contains_cnt, player_move = 0, -1
    for t,v in rps_map.items():
        if t in msg.content.lower():
            contains_cnt += 1
            player_move = v

    if contains_cnt != 1:
        return

    # check whether we're playing or not
    if 'rps_id' not in chan_info:
        return
    if chan_info['rps_id'] != author.id:
        return

    bot_move = random.randint(0, 2)
    await channel.send(RPS_BOTS_MOVE.format(rps_revmap[bot_move]))

    score = rps_matrix[bot_move][player_move]

    if score > 0:
        await channel.send(RPS_USER)
        channel_add_mood(channel.id, -random.randint(1, 10))
    elif score == 0:
        await channel.send(RPS_DRAW)
    elif score < 0:
        await channel.send(RPS_BOT)
        channel_add_mood(channel.id, random.randint(1, 10))

    del chan_info['rps_id']

# sets a unload timer
def schedule_unload(id):
    if id == 0:
        return

    timer = channel_timers.pop(id, None)

    if timer is not None:
        timer.cancel()

    timer = threading.Timer(60 * 10, unload_channel, (id,))
    timer.start()
    channel_timers[id] = timer

def channel_lock(id):
    lock = channel_locks.pop(id, None)
    if lock is None:
        lock = threading.Lock()
        channel_locks[id] = lock
    return lock

# unloads a channel from memory
def unload_channel(id):
    if id == 0:
        return

    with global_c_lock:
        channel_timers.pop(id, None)
        # bail the heck outta here while we haven't corrupted everything
        if channel_lock(id).locked():
            return

        save_channel(id)
        del channels[id]['model']
        del channels[id]
        gc.collect()

        print('X-X', id) # unloaded

# saves channel settings and model
def save_channel(id):
    global channels

    with channel_lock(id):
        with gzip.open(f'channels/{id}.json.gz', 'wb') as f:
            to_jsonify = channels[id].copy()

            if to_jsonify['model'] is not None:
                to_jsonify['model'] = to_jsonify['model'].to_dict()

            f.write(json.dumps(to_jsonify).encode('utf-8'))

    print('<--', id) # dumped

def channel_exists(id):
    return os.path.exists(f'channels/{str(id)}.json.gz')

# loads channel settings and model
async def load_channel(id, channel):
    global channels

    # prevent corruption
    with global_c_lock:
        with channel_lock(id):
            # abort any unloading timers
            timer = channel_timers.pop(id, None)
            if timer is not None:
                timer.cancel()

            try:
                with gzip.open(f'channels/{id}.json.gz', 'rb') as f:
                    jsonified = json.loads(f.read().decode('utf-8'))

                    if jsonified['model'] != None:
                        jsonified['model'] = markovify.NewlineText.from_dict(jsonified['model'])
                    channels[id] = jsonified

            except (JSONDecodeError, OSError, zlib.error) as ex:
                # notify me
                await (await bot.application_info()).owner.send(f':x: **Model failed to load ({id})**')

                print('!!!', id) # failed to load
                if channel is not None:
                    await channel.send(JSON_DEC_FAILURE)
                return

            # add fields that appeared in newer versions of the bot
            chan_info = channels[id]
            new_fields = {
                'total_msgs':         0,
                'uwumode':            False,
                'ustats':             {},
                'next_gen_milestone': chan_info['autorate'],
                'mood':               50,
                'actions':            True,
                'ignore_bots':        True,
                'remove_mentions':    False
            }
            for k, v in new_fields.items():
                if k not in chan_info:
                    chan_info[k] = v

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

    # remove mentions (incl. channels, etc.) from messages generated from the global model and/or if requested
    if id == 0 or channels[act_id]['remove_mentions']:
        generated_msg = re.sub('<(@|&|#)!*[0-9]*>', '**[mention removed]**', generated_msg)

    print(' G ', id)

    return generated_msg.replace('%', '"')

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

# trains a model on a message
def train(id, text_list):
    def _do_train(id, text_list):
        try:
            with channel_lock(id):
                chan_info = channels[id]

                for text in text_list:
                    # join period-separated sentences by a new line
                    # and replace " with %
                    text = '\n'.join(text.split('.'))
                    text = text.replace('"', '%')

                    # create a new model if it doesn't exist
                    if chan_info['model'] == None:
                        chan_info['model'] = markovify.NewlineText(text)

                    # create a model with this message and combine it with the already existing one ("train" the model)
                    new_model = markovify.NewlineText(text)
                    chan_info['model'] = markovify.combine([channels[id]['model'], new_model])

                # increment the number of collected messages
                chan_info['collected'] += len(text_list)

                print(' + ', id)
                        
        except Exception as ex:
            traceback.print_exception(type(ex), ex, ex.__traceback__)

    Thread(target=_do_train, args=(id, text_list), name='Training thread').start()

# trains a channel on previous messages (takes some time, meant to be run in a separate thread)
def train_on_prev(chan_id, limit):
    msgs = get_msgs(chan_id, 0, limit)
    train(chan_id, [m['content'] for m in msgs])





def on_bot_exit():
    # Thank you
    # I'll say goodbye soon
    # Though it's the end of the world
    # Don't blame yourself, no
    print(f'<-> SAVING {len(channels)} CHANNELS, DON\'T INTERRUPT')
    with global_c_lock:
        for chan_id in channels:
            save_channel(chan_id)
    exit()

def bot_presence_thread():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    while True:
        loop.run_until_complete(upd_bot_presence())
        time.sleep(10)

async def upd_bot_presence():
    await bot.change_presence(activity=discord.Activity(
        type=discord.ActivityType.playing,
        name=f'around in {str(len(bot.guilds))} servers | "{bot.command_prefix}help" for help'
    ))

@bot.event
async def on_ready():
    global process

    await load_channel(0, None)
    Thread(target=bot_presence_thread, name='Presence updater').start()
    atexit.register(on_bot_exit)
    process = psutil.Process(os.getpid())

    print('Everything OK!')

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

def get_key(d, val): 
    for k, v in d.items():
        if v == val:
            return k
    return None

@bot.command(pass_context=True, name='rps')
async def scoreboard_cmd(ctx):
    await channel_game_start(ctx.channel, ctx.message.author)

@bot.command(pass_context=True, name='scoreboard')
async def scoreboard_cmd(ctx):
    channel_info = channels[ctx.message.channel.id]
    ustats = channel_info['ustats']
    # get 10 biggest values and form a scoreboard embed
    top = {k: v for k, v in sorted(ustats.items(), key=lambda item: item[1], reverse=True)}
    top_u, top_c = list(top.keys())[:10], list(top.values())[:10]

    embed = discord.Embed(title='Deuterium scoreboard', color=EMBED_COLOR)
    for i in range(len(top_u)):
        u = top_u[i]
        c = top_c[i]
        embed.add_field(inline=False, name=f'#{str(i+1)}', value=f'<@{str(u)}> - **{str(c)}** messages')

    if len(top_u) == 0:
        embed.add_field(inline=False, name=f'Empty', value='Enable message collection (`!!d collect yes`) and talk for a while')
    await ctx.send(embed=embed)

@bot.command(pass_context=True, name='status')
async def status_cmd(ctx):
    id = ctx.message.channel.id
    chan_info = channels[ctx.message.channel.id]
    autorate = chan_info['autorate']
    embed = discord.Embed(title='Deuterium status', color=EMBED_COLOR)
    embed.add_field(name=':books: Learning messages in this channel',                        value='yes' if chan_info['collect'] else 'no')
    embed.add_field(name=':1234: Total messages trained on',                                 value=str(chan_info['collected']))
    embed.add_field(name=':rocket: Contributing to the global model',                        value='yes' if chan_info['gcollect'] else 'no')
    embed.add_field(name=':1234: Messages added to the global model by this channel',        value=str(chan_info['gcollected']))
    embed.add_field(name=':1234: Messages added to the global model by everyone everywhere', value=str(channels[0]['collected']))
    embed.add_field(name=':new: Automatically sending messages after each',                  value=str('[disabled]' if autorate == 0 else autorate))

    embed.add_field(name=':sparkles: Mood', value=f'{channel_mood(id)["title"]} ({chan_info["mood"]}%)')
    embed.add_field(name=':eyes: UwU mode', value='enabled' if chan_info['uwumode'] else 'disabled')

    await ctx.send(embed=embed)

@bot.command(pass_context=True, name='stats')
async def stats_cmd(ctx):
    global process

    embed = discord.Embed(title='Deuterium stats', color=EMBED_COLOR)

    chan_cnt  = len(os.listdir('channels')) - 1 # descrement because of the global model
    chan_size = 0
    for path, _, files in os.walk('channels'):
        for f in files:
            chan_size += os.path.getsize(os.path.join(path, f))

    embed.add_field(name='Total model count', value=f'**`{str(chan_cnt)}`**')
    embed.add_field(name='Disk space used',   value=f'**`{str(chan_size // (1024**2))} MiB`**')
    embed.add_field(name='Loaded models',     value=f'**`{str(len(channels))}`**')

    embed.add_field(name='RAM used',   value=f'**`{str(process.memory_info().rss // (1024**2))} MiB`**')
    embed.add_field(name='CPU used',   value=f'**`{str(process.cpu_percent())}%`**')
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
    if not has_admin(ctx):
        await ctx.send(ADMINISTRATIVE_C)
        return

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

async def yes_no_cmd(ctx, prop, admin, args):
    if admin and not has_admin(ctx):
        await ctx.send(ADMINISTRATIVE_C)
        return

    if len(args) != 1:
        await ctx.send(ONE_ARG)
        return

    inp = args[0]
    if inp not in ['yes', 'no']:
        await ctx.send(INVALID_ARG)
    else:
        channels[ctx.message.channel.id][prop] = True if inp == 'yes' else False
        await ctx.send(SETS_UPD)

@bot.command(pass_context=True, name='actions')
async def actions_cmd(ctx, *args):
    await yes_no_cmd(ctx, 'actions', True, args)

@bot.command(pass_context=True, name='collect')
async def collect_cmd(ctx, *args):
    await yes_no_cmd(ctx, 'collect', True, args)

@bot.command(pass_context=True, name='gcollect')
async def gcollect_cmd(ctx, *args):
    await yes_no_cmd(ctx, 'gcollect', True, args)

@bot.command(pass_context=True, name='ignore_bots')
async def ignore_bots_cmd(ctx, *args):
    await yes_no_cmd(ctx, 'ignore_bots', True, args)

@bot.command(pass_context=True, name='remove_mentions')
async def remove_mentions_cmd(ctx, *args):
    await yes_no_cmd(ctx, 'remove_mentions', True, args)

@bot.command(pass_context=True, name='reset')
async def reset_cmd(ctx):
    if not has_admin(ctx):
        await ctx.send(ADMINISTRATIVE_C)
        return

    chan_info = channels[ctx.message.channel.id]
    chan_info['model'] = None
    chan_info['collected'] = 0
    chan_info['total_msgs'] = 0
    chan_info['next_gen_milestone'] = 0
    chan_info['mood'] = 50
    await ctx.send(RESET_SUCCESS)

@bot.command(pass_context=True, name='autorate')
async def autorate_cmd(ctx, *args):
    if not has_admin(ctx):
        await ctx.send(ADMINISTRATIVE_C)
        return
        
    if len(args) != 1:
        await ctx.send(ONE_ARG)
        return

    chan_info = channels[ctx.message.channel.id]

    autorate = args[0]
    try:
        autorate = int(autorate)
        if autorate < 0:
            await ctx.send(INVALID_ARG)
            return
        chan_info['autorate'] = autorate
        await ctx.send(AUTOGEN_DISABLED if autorate == 0 else (AUTOGEN_SETTO.format(str(autorate))))
    except ValueError:
        await ctx.send(INVALID_ARG)

@bot.event
async def on_command_error(ctx, ex: Exception):
    if type(ex) is discord.ext.commands.errors.CommandNotFound:
        await ctx.send(UNKNOWN_CMD)
    elif type(ex) is Forbidden:
        pass # ignore
    else:
        print(ex)
        await exception_handler(type(ex), ex, ex.__traceback__)



# we need this because the bot has to train its message generation models
@bot.event
async def on_message(msg: discord.Message):
    global channels
    channel = msg.channel

    # ignore ourselves
    if msg.author == bot.user:
        return

    # don't react to DMs
    if msg.guild == None:
        await channel.send(':x: **This bot only works in servers**')
        return

    # don't react to empty messages (if it's just a picture, or audio, etc.)
    if len(msg.content) == 0:
        return

    if msg.content == '/d help':
        await msg.channel.send(NEW_PREFIX)
        return

    # load channel settings and model from the disk if available and needed
    chan_id = int(channel.id)
    if channel_exists(chan_id) and chan_id not in channels:
        await load_channel(chan_id, channel)

    # create a new channel object if it doesn't exist
    if chan_id not in channels:
        chan_info = {
            'model':     None,
            'collect':   True,  'collected':  0,
            'autorate':  20,    'total_msgs': 0,
            'gcollect':  False, 'gcollected': 0,
            'uwumode':   False, 'ustats':     {},

            'next_gen_milestone': 20,
            'mood':               50,
            'actions':            True,
            'ignore_bots':        True,
            'remove_mentions':    False
        }
        channels[chan_id] = chan_info
    
    chan_info = channels[chan_id]
    chan_info['total_msgs'] += 1

    # unload the channel in 10s
    try:
        timer = channel_timers[chan_id]
        timer.cancel()
    except: pass
    timer = threading.Timer(30, unload_channel, [chan_id])
    #timer.start()
    channel_timers[chan_id] = timer

    # (optionally) ignore bots
    if msg.author.bot and not chan_info['ignore_bots']:
        return

    # check if it's a command
    if msg.content.startswith(bot.command_prefix):
        # process the command and return
        await bot.process_commands(msg)
        return

    # it's an ordinary message and not a command
    # train on this message if allowed
    if chan_info['collect']:
        train(chan_id, [msg.content])

        # add a score to the user
        ustats = chan_info['ustats']
        author = str(msg.author.id)
        if author not in ustats:
            ustats[author] = 0
        ustats[author] += 1

    # train the global model if allowed
    if chan_info['gcollect']:
        train(0, [msg.content])
        chan_info['gcollected'] += 1

    # react to mentions
    if bot.user in msg.mentions:
        m = await generate_channel(chan_id, chan_id)
        await channel.send(f'{msg.author.mention} {m}')
        return

    # mood logic
    channel_add_mood(chan_id, random.randint(-1, 2))
    mood = channel_mood(chan_id)

    if chan_info['actions']:
        # suggest playing a game
        if random.random() <= mood['suggest_chance']:
            await channel.send(RPS_SUGGESTION.format(msg.author.mention))
            return

        # give a treat to the sender
        if random.random() <= mood['treat_chance']:
            await channel.send(f'Hey {msg.author.mention}! Have a {random.choice(mood["treats"])}')
            return

    # process the game
    await channel_game_process(channel, msg, msg.author)

    # generate a message automatically
    rate = chan_info['autorate']
    if rate > 0 and chan_info['total_msgs'] >= chan_info['next_gen_milestone']:
        # set next milestone
        qrate = rate // 4
        chan_info['next_gen_milestone'] = chan_info['total_msgs'] + random.randint(rate - qrate, rate + qrate)

        # generate a message
        await generate_channel_threaded(channel)
        return

async def exception_handler(exctype, excvalue, exctraceback):
    if exctype is KeyboardInterrupt or exctype is SystemExit:
        sys.__excepthook__(exctype, excvalue, exctraceback)

    traceback.print_exception(exctype, excvalue, exctraceback)
    await (await bot.application_info()).owner.send(f':x: **Exception**\n```\n{str(excvalue)}\n```')

def exception_wrapper(exctype, excvalue, exctraceback):
    loop = asyncio.get_event_loop()
    loop.run_until_complete(exception_handler)

sys.excepthook = exception_wrapper

bot.run(TOKEN)
