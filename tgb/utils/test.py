'''
## FLIGHT DATASET
def csv_to_pd_data(
    fname: str,
) -> pd.DataFrame:
    r"""
    currently used by tgbl-flight dataset
    convert the raw .csv data to pandas dataframe and numpy array
    input .csv file format should be: timestamp, node u, node v, attributes
    Args:
        fname: the path to the raw data
    """
    
    feat_size = 16
    num_lines = 10 #sum(1 for line in open(fname)) - 1
    print("number of lines counted", num_lines)
    u_list = np.zeros(num_lines)
    i_list = np.zeros(num_lines)
    ts_list = np.zeros(num_lines)
    label_list = np.zeros(num_lines)
    feat_l = np.zeros((num_lines, feat_size))
    idx_list = np.zeros(num_lines)
    w_list = np.zeros(num_lines)
    print("numpy allocated")
    node_ids = {}
    unique_id = 0
    ts_format = None

    with open(fname, "r") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        idx = 0
        #'day','src','dst','callsign','typecode'
        for row in tqdm(csv_reader):
            if idx == 0:
                idx += 1
                continue
            else:
                ts = row[0]
                if ts_format is None:
                    if (ts.isdigit()):
                        ts_format = True
                    else:
                        ts_format = False
                
                if ts_format:
                    ts = float(int(ts)) #unix timestamp already
                else:
                    #convert to unix timestamp
                    TIME_FORMAT = "%Y-%m-%d"
                    date_cur = datetime.datetime.strptime(ts, TIME_FORMAT)
                    ts = float(date_cur.timestamp())
                    # TIME_FORMAT = "%Y-%m-%d" # 2019-01-01
                    # date_cur  = date.fromisoformat(ts)
                    # dt = datetime.datetime.combine(date_cur, datetime.datetime.min.time())
                    # dt = dt.replace(tzinfo=datetime.timezone.edt)
                    # ts = float(dt.timestamp())


                src = row[1]
                dst = row[2]

                # 'callsign' has max size 8, can be 4, 5, 6, or 7
                # 'typecode' has max size 8
                # use ! as padding

                # pad row[3] to size 7
                if len(row[3]) == 0:
                    row[3] = "!!!!!!!!"
                while len(row[3]) < 8:
                    row[3] += "!"

                # pad row[4] to size 4
                if len(row[4]) == 0:
                    row[4] = "!!!!!!!!"
                while len(row[4]) < 8:
                    row[4] += "!"
                if len(row[4]) > 8:
                    row[4] = "!!!!!!!!"

                feat_str = row[3] + row[4]

                if src not in node_ids:
                    node_ids[src] = unique_id
                    unique_id += 1
                if dst not in node_ids:
                    node_ids[dst] = unique_id
                    unique_id += 1
                u = node_ids[src]
                i = node_ids[dst]
                u_list[idx - 1] = u
                i_list[idx - 1] = i
                ts_list[idx - 1] = ts
                idx_list[idx - 1] = idx
                w_list[idx - 1] = float(1)
                feat_l[idx - 1] = convert_str2int(feat_str)
                idx += 1
    return (
        pd.DataFrame(
            {
                "u": u_list,
                "i": i_list,
                "ts": ts_list,
                "label": label_list,
                "idx": idx_list,
                "w": w_list,
            }
        ),
        feat_l,
        node_ids,
    )

### COIN DATASET
feat_size = 1
    num_lines = sum(1 for line in open(fname)) - 1
    print("number of lines counted", num_lines)
    u_list = np.zeros(num_lines)
    i_list = np.zeros(num_lines)
    ts_list = np.zeros(num_lines)
    label_list = np.zeros(num_lines)
    feat_l = np.zeros((num_lines, feat_size))
    idx_list = np.zeros(num_lines)
    w_list = np.zeros(num_lines)
    print("numpy allocated")
    node_ids = {}
    unique_id = 0

    with open(fname, "r") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        idx = 0
        # time,src,dst,weight
        # 1648811421,0x27cbb0e6885ccb1db2dab7c2314131c94795fbef,0x8426a27add8dca73548f012d92c7f8f4bbd42a3e,800.0
        for row in tqdm(csv_reader):
            if idx == 0:
                idx += 1
                continue
            else:
                ts = int(row[0])
                src = row[1]
                dst = row[2]

                if src not in node_ids:
                    node_ids[src] = unique_id
                    unique_id += 1
                if dst not in node_ids:
                    node_ids[dst] = unique_id
                    unique_id += 1

                w = float(row[3])
                if w == 0:
                    w = 1

                u = node_ids[src]
                i = node_ids[dst]
                u_list[idx - 1] = u
                i_list[idx - 1] = i
                ts_list[idx - 1] = ts
                idx_list[idx - 1] = idx
                w_list[idx - 1] = w
                feat_l[idx - 1] = np.zeros(feat_size)
                idx += 1

    #! normalize by log 2 for stablecoin
    w_list = np.log2(w_list)

    return (
        pd.DataFrame(
            {
                "u": u_list,
                "i": i_list,
                "ts": ts_list,
                "label": label_list,
                "idx": idx_list,
                "w": w_list,
            }
        ),
        feat_l,
        node_ids,
    )

### COMMENT DATASET
def csv_to_pd_data_rc(
    fname: str,
) -> pd.DataFrame:
    r"""
    currently used by redditcomments dataset
    convert the raw .csv data to pandas dataframe and numpy array
    input .csv file format should be: timestamp, node u, node v, attributes
    Args:
        fname: the path to the raw data
    """
    feat_size = 2  # 1 for subreddit, 1 for num words
    num_lines = sum(1 for line in open(fname)) - 1
    #print("number of lines counted", num_lines)
    print("there are ", num_lines, " lines in the raw data")
    u_list = np.zeros(num_lines)
    i_list = np.zeros(num_lines)
    ts_list = np.zeros(num_lines)
    label_list = np.zeros(num_lines)
    feat_l = np.zeros((num_lines, feat_size))
    idx_list = np.zeros(num_lines)
    w_list = np.zeros(num_lines)
    node_ids = {}

    unique_id = 0
    max_words = 5000  # counted form statistics

    with open(fname, "r") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        idx = 0
        # ['ts', 'src', 'dst', 'subreddit', 'num_words', 'score']
        for row in tqdm(csv_reader):
            if idx == 0:
                idx += 1
                continue
            else:
                ts = int(row[0])
                src = row[1]
                dst = row[2]
                num_words = int(row[3]) / max_words  # int number, normalize to [0,1]
                score = int(row[4])  # int number

                # reindexing node and subreddits
                if src not in node_ids:
                    node_ids[src] = unique_id
                    unique_id += 1
                if dst not in node_ids:
                    node_ids[dst] = unique_id
                    unique_id += 1
                w = float(score)
                u = node_ids[src]
                i = node_ids[dst]
                u_list[idx - 1] = u
                i_list[idx - 1] = i
                ts_list[idx - 1] = ts
                idx_list[idx - 1] = idx
                w_list[idx - 1] = w
                feat_l[idx - 1] = np.array([num_words])
                idx += 1
    print("there are ", len(node_ids), " unique nodes")

    return (
        pd.DataFrame(
            {
                "u": u_list,
                "i": i_list,
                "ts": ts_list,
                "label": label_list,
                "idx": idx_list,
                "w": w_list,
            }
        ),
        feat_l,
        node_ids,
    )

'''