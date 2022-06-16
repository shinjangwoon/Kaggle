game_index_dic = {"solo": 1, "solo-fpp": 1, 'normal-solo': 1, "normal-solo-fpp": 1,
                  "duo": 2, "duo-fpp": 2, 'normal-duo': 2, "normal-duo-fpp": 2,
                  "squad": 3, "squad-fpp": 3, 'normal-squad': 3, "normal-squad-fpp": 3,
                  "crashfpp": 4, "crashtpp": 4,
                  "flarefpp": 5, "flaretpp": 5
                  }

game_name_dic = {"solo": "solo", "solo-fpp": "solo", 'normal-solo': "solo", "normal-solo-fpp": "solo",
                 "duo": "duo", "duo-fpp": "duo", 'normal-duo': "duo", "normal-duo-fpp": "duo",
                 "squad": "squad", "squad-fpp": "squad", 'normal-squad': "squad", "normal-squad-fpp": "squad",
                 "crashfpp": "crash", "crashtpp": "crash",
                 "flarefpp": "flare", "flaretpp": "flare"
                 }

train['matchTypeName'] = train['matchType'].apply(lambda x: game_name_dic[x])
train['matchType'] = train['matchType'].apply(lambda x: game_index_dic[x])

# 그룹 멤버 = match id가 같다면 같은 게임을 진행함, group id 가 같다면 같은 그룹이다.
# 이벤트 모드에서 팀 게임을 구분하지 못하는게 아쉬워서 다른 기준을 세워 보았습니다.
team_member = train.groupby(['matchId', 'groupId']).size().to_frame('teamSize')
train = train.merge(team_member, how='left', on=['matchId', 'groupId'])

# Solos : 1 player
# Duos : 2 players
# Squad : 1 to 4 players
# Crash : 1 to 4 players
# Flaredrop : 1 to 4 players
# 어떤 모드에서든 팀 멤버는 4명을 넘을 수 없습니다.
# 혹시 아니라면 알려주세요.
size_dic = {1: 1, 2: 2, 3: 4, 4: 4, 5: 4}

train['matchTeamSize'] = train['teamSize'] / \
    train['matchType'].apply(lambda x: size_dic[x])

# 숙련도(headshot은 게임에 능숙한 사람이 높은 점수를 얻을 것이다, longestkill의 경우도 게임에 익숙한 사람일 것이다.)
# team kill의 경우 게임에서 이기고자 하는 사람이 아니기 때문에 이 값을 뺐습니다.
# 한명도 죽이지 못한 경우에 수식이 돌아가지 않아서 1의 값을 추가했습니다.
train['skillfull'] = train['headshotKills'] + 0.01 * \
    train['longestKill'] - train['teamKills']/(train['kills']+1)

train['hsRatio'] = train['headshotKills'] / train['kills']
train['hsRatio'].fillna(0, inplace=True)

# 1명을 headshot으로 죽였는데 전체 게임에서 1명만 죽였을 경우 숙련된 게이머라고 보기 어렵습니다.


def transform_hsRatio(x):
    if x == 1 or x == 0:
        return 0.5
    else:
        return x


train['hsRatio'] = train['hsRatio'].apply(transform_hsRatio)

# 1게임 안에서 이동한 거리를 비교하기 위해 match Duration으로 나눴습니다.
train['distance'] = (train['walkDistance'] + train['rideDistance'] +
                     train['swimDistance'])/train['matchDuration']

# 아이템 역시 한 게임 안에서 사용한 양을 비교하기 위해 나눴습니다.
# 사실 이동한 거리로 계산해야 하는지 고민이듭니다.
train['itemsRatio'] = train['heals'] + train['boosts'] / train['matchDuration']
train['itemsRatio'].fillna(0, inplace=True)
train['itemsRatio'].replace(np.inf, 0, inplace=True)

train['killsRatio'] = train['kills'] + train['killStreaks'] + \
    train['roadKills'] + train['headshotKills'] / train['matchDuration']**0.1
train['killsRatio'].fillna(0, inplace=True)
train['killsRatio'].replace(np.inf, 0, inplace=True)
