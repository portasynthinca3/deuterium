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
Please consider supporting us (`/d support`).
'''

ANNOUNCEMENT = '''
I have moved to a new server with an AMD EPYC server CPU! This means that I'll probably be able to implement a new message generation technique
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
JSON_DEC_FAILURE = ':x: **The model failed to load, probably because it became corrupt in a recent change to how the bot stores channel info. Please either do a `/d reset` or join our support server (<https://discord.gg/N52uWgD>). If this problem stops occuring in a minute or so, that\'s probably because I restored a backup.**'
TOO_MANY_MSGS    = ':x: **Too many messages requested (10 max)**'
ADMINISTRATIVE_C = ':x: **This command is administrative. Only people with the "administrator" privilege can execute it**'
RESET_SUCCESS    = '**Successfully reset training data for this channel** :white_check_mark:'
AUTOGEN_DISABLED = '**Auto-generation disabled** :white_check_mark:'
AUTOGEN_SETTO    = '**Auto-generation rate set to {0}** :white_check_mark:'
CREATOR_ONLY     = ':x: **This command can only be executed by the creator of the bot**'
MADE_NEURAL      = '**This channel is now using a neural network** :white_check_mark:'

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

import numpy as np
from keras.utils import np_utils
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model as keras_load_model
from keras.models import Sequential
from keras.layers import Dense, LSTM, Activation, Dropout

# prepare constant embeds
EMBED_COLOR = 0xe6f916

HELP_CMDS_REGULAR = {
    'help':                 'sends this message',
    'status':               'tells you the current settings and stats',
    'stats':                'global Deuterium stats',
    'train <count>':        'trains the model using the last <count> messages in this channel',
    'gen':                  'generates a message immediately',
    'gen <count>':          'generates <count> messages immediately',
    'gen #channel-mention': 'generates a message using the mentioned channel model',
    'ggen':                 'generates a message immediately (using the global model)',
    'support':              'ways to support this project',
    'privacy':              'our privacy policy',
    'uwu <enable/disable>': 'enable/disable the UwU mode (don\'t.)',
    'info':                 'who created this?',
    'scoreboard':           'shows top 10 most active users',
    #'gscoreboard':          'shows top 10 most active users in the global model'
}
HELP_CMDS_ADMIN = {
    'collect <yes/no>':    'allows or denies training using new messages',
    'gcollect <yes/no>':   'allows or denies training the global model using new messages',
    'reset':               'resets the training model',
    'autorate <rate>':     'sets the automatic generation rate (one bot message per each <rate> normal ones)',
    #'arole @role-mention': 'grants administrative permissions to the mentioned role. Only one role can have them at a time'
}

HELP = discord.Embed(title='Deuterium commands', color=EMBED_COLOR)
HELP.add_field(inline=False, name='Latest Announcement', value=ANNOUNCEMENT)

HELP.add_field(inline=False, name='*All comands start with `/d `*', value='*ex.: `/d help`*')

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
- if the message starts with `/d `, the bot treats it as a command and doesn't store it
- else, if this channel has its "collect" setting set to "yes", it trains the model on this message and saves the said model do disk
- if this channel has its "global collect" setting set to "yes", it trains the global model on this message and saves the said model do disk''')
PRIVACY.add_field(inline=False, name='4. DATA STORAGE', value='''
Deuterium stores the following data:
- Channel settings and statistics (is message collection allowed, the total number of collected messages, etc.). This data can be viewed using the `/d status` command
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
If you wish to reset the local model, you may use the `/d reset` command.
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

# some global variables
SELF_ID = None
channels = {}
chanel_timers = {}
channel_limits = {}
channel_locks = {}
global_c_lock = threading.Lock()
process = None

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

    # prevent corruption
    with global_c_lock:
        save_channel(id)
        del channels[id]['model']
        del channels[id]
        gc.collect()

        print('X-X', id) # unloaded

        chanel_timers.pop(id, None)

# saves channel settings and model
def save_channel(id):
    global channels

    with channel_lock(id):
        with gzip.open(f'channels/{id}.json.gz', 'wb') as f:
            to_jsonify = channels[id].copy()

            if to_jsonify['model'] is not None:
                to_jsonify['model'] = to_jsonify['model'].to_dict()

            # save the neural model separately
            n_model = to_jsonify['n_model']
            if n_model is not None:
                n_model.save(f'neural_channels/{id}.tf', save_format='tf')
                print('<-N', id) # dumped neural
                del to_jsonify['n_model']

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
            timer = chanel_timers.pop(id, None)
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
                'total_msgs': 0,
                'uwumode':    False,
                'ustats':     {},

                'is_neural':  False,
                'n_model':    None
            }
            for k,v in new_fields.items():
                if k not in chan_info:
                    chan_info[k] = v

            # load the neural model (if it exists)
            if channels[id]['is_neural']:
                channels[id]['n_model'] = keras_load_model(f'neural_channels/{id}.tf')
                print('N->', id) # loaded neural

            print('-->', id) # loaded

# generates a message
async def generate_channel(id, act_id):
    global channels

    if channel_exists(id) and id not in channels:
        await load_channel(id, None)

    if channels[id]['is_neural']: # neural net
        chan_info = channels[id]
        final_buf = chan_info['n_buffer']
        seq_len   = chan_info['n_seq_len']
        to_num    = chan_info['n_convert_to']
        from_num  = chan_info['n_convert_from']
        model     = chan_info['n_model']

        # start with the start indicator
        generated_msg = '\1' * seq_len

        # stop at the stop indicator (limit to 50 chars)
        last_char, cnt = '', 0
        while last_char != '\2' and cnt < 50:
            encoded = np.array([to_num[c] for c in generated_msg][-seq_len:])
            encoded = np_utils.to_categorical(encoded, num_classes=len(to_num))
            encoded = encoded.reshape(1, seq_len, len(to_num))

            # generate a char and convert it to an actual char
            out_tensor = np.argmax(model.predict(encoded), axis=-1)
            out_idx = out_tensor[0]
            last_char = from_num[str(out_idx)]

            generated_msg += last_char
            cnt += 1

        generated_msg = generated_msg.replace('\n', ' ').strip().replace('\1', '').replace('\2', '')
        if generated_msg == '':
            return GEN_FAILURE

        final_buf += ' ' + generated_msg
        chan_info['n_buffer'] = final_buf[-seq_len:]

    else: # markov chain
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

# trains a model on a message
def train(id, text_list):
    def _do_train(id, text_list):
        try:
            with channel_lock(id):
                chan_info = channels[id]
                if chan_info['is_neural']: # neural net
                    final_buf = chan_info['n_buffer'] + ''.join([f'\1{s}\2' for s in text_list])
                    seq_len   = chan_info['n_seq_len']
                    to_num    = chan_info['n_convert_to']

                    # make a lsit of sequences
                    seqs = []
                    for i in range(seq_len, len(final_buf)):
                        seqs.append(final_buf[i-seq_len : i+1])
                    seqs = [[float(to_num[c]) if c in to_num else to_num[' '] for c in s] for s in seqs]

                    # cut the buffer
                    chan_info['n_buffer'] = final_buf[-seq_len-1:]

                    # make a list of X|->Y associations
                    seqs = np.array(seqs)
                    X, Y = seqs[:,:-1], seqs[:,-1]
                    X = np.array([np_utils.to_categorical(s, num_classes=len(to_num)) for s in X])
                    Y =           np_utils.to_categorical(Y, num_classes=len(to_num))

                    # train the model
                    chan_info['n_model'].fit(X, Y, epochs=25, verbose=1)

                else: # Markov chain
                    for text in text_list:
                        # join period-separated sentences by a new line
                        text = '\n'.join(text.split('.'))

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





bot = Bot(command_prefix='/d ', help_command=None)

def on_bot_exit():
    # Thank you
    # I'll say goodbye soon
    # Though it's the end of the world
    # Don't blame yourself, no
    print('<-> SAVING CHANNELS, DON\'T INTERRUPT')
    with global_c_lock:
        for chan_id in channels:
            save_channel(chan_id)

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

@bot.command(pass_context=True, name='make_neural')
async def info_cmd(ctx):
    if not await bot.is_owner(ctx.message.author):
        await ctx.send(CREATOR_ONLY)
        return

    # Erase old data
    channel_info = channels[ctx.message.channel.id]
    channel_info['is_neural'] = True
    channel_info['collected'] = 0
    channel_info['model']     = None
    channel_info['ustats']    = {}

    # Default NN settigs
    seq_len = 25
    channel_info['n_seq_len'] = seq_len
    channel_info['n_buffer']  = '\1' * seq_len

    # Create conversion dicts
    chars = list('\1\2' + string.printable)
    channel_info['n_convert_to']   = {chars[i]:i for i in range(len(chars))}
    channel_info['n_convert_from'] = {str(v):k for k,v in channel_info['n_convert_to'].items()}

    # Create the model itself
    model = Sequential()
    channel_info['n_model'] = model
    model.add(LSTM(16, input_shape=(seq_len, len(chars)), return_sequences=False))
    model.add(Dropout(0.05))
    model.add(Dense(len(chars), kernel_initializer='normal', activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    await ctx.send(MADE_NEURAL)

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
        embed.add_field(inline=False, name=f'Empty', value='Enable message collection (`/d collect yes`) and talk for a while')
    await ctx.send(embed=embed)

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

    if chan_info['is_neural']:
        embed.add_field(name='Neural :brain:', value='This model is using a neural network instead of a Markov chain.')
        embed.add_field(name='Neural Buffer',  value=f'`{chan_info["n_buffer"]}`')
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
    for path, _, files in os.walk('neural_channels'):
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
    chan_id = int(channel.id)
    if channel_exists(chan_id) and chan_id not in channels:
        await load_channel(chan_id, channel)

    # create a new channel object if it doesn't exist
    if chan_id not in channels:
        chan_info = {
            'model':     None,
            'is_neural': False, 'n_model':    None,
            'collect':   True,  'collected':  0,
            'autorate':  20,    'total_msgs': 0,
            'gcollect':  False, 'gcollected': 0,
            'uwumode':   False, 'ustats':     {}
        }
        channels[chan_id] = chan_info
    
    chan_info = channels[chan_id]

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

    # generate a message if needed
    if chan_info['autorate'] > 0 and chan_info['total_msgs'] % chan_info['autorate'] == 0:
        await generate_channel_threaded(channel)

    chan_info['total_msgs'] += 1

    # unload the channel in a while
    if not chan_info['is_neural']:
        schedule_unload(chan_id)

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
