import tensorflow as tf
import pandas as pd
import youtube_dl
import os
from pydub import AudioSegment

"""
path_features = TFrecords directory
path_videos = directory in which you want to save audio from youtube

CSV file is created in the script directory

"""
path_features = "C:\\Users\\kowal\\Desktop\\Nowy folder\\audioset_v1_embeddings\\eval"
path_videos = "C:\\Users\\kowal\\Desktop\\Pliki"

ydl_opts = {
    'format': 'bestaudio/best',
    'postprocessors': [{
        'key': 'FFmpegExtractAudio',
        'preferredcodec': 'wav',
        'preferredquality': '192',
    }],
    'outtmpl': path_videos + '/%(title)s.%(ext)s',
}
dict = {}
i = 0

if __name__ == '__main__':
    filenames = [os.path.join(path_features,file) for file in os.listdir(path_features)] #scan directory and add every file
    raw_dataset = tf.data.TFRecordDataset(filenames)    #load raw data
    for raw_record in raw_dataset:
        example = tf.train.SequenceExample()            #load it as sequence example protocol buffer
        example.ParseFromString(raw_record.numpy())     #parse it

        try:
            with youtube_dl.YoutubeDL(ydl_opts) as ydl:                                          #download yt video
                info_dict = ydl.extract_info(os.path.join('http://www.youtube.com/watch?v=' +
                                           str(example.context.feature["video_id"].bytes_list.value[0].decode())), download=True)

                video_title = info_dict.get('title', None)
                audio = AudioSegment.from_file(os.path.join(path_videos, video_title + ".wav")) #cut video and save only
                                                                                                #the right part of it
                t1 = example.context.feature["start_time_seconds"].float_list.value[0] * 1000
                t2 = example.context.feature["end_time_seconds"].float_list.value[0] * 1000
                audio = audio[t1:t2]
                audio.export(os.path.join(path_videos, video_title + ".wav"), format="wav")

        except:
            continue


        #Dictionary I use to creat dataframe
        dict[i]  = {"video_id" : example.context.feature["video_id"].bytes_list.value[0].decode(),
                   "start_time_seconds" : example.context.feature["start_time_seconds"].float_list.value[0],
                   "end_time_seconds" : example.context.feature["end_time_seconds"].float_list.value[0],
                   "path" : os.path.join(path_videos, video_title + ".wav")}
        i += 1

    #Creat df and save it in CSV
    df = pd.DataFrame.from_dict(dict, "index")
    df.to_csv(r'TFrecord_to_csv.csv', index = False)

