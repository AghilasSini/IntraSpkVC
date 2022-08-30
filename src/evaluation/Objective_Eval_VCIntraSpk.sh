# input
# Stimuli Files 
#   - 
data_dir="/mnt/mydrive/home/aghilas/Workspace/Experiments/SpeakerCoding/Resemblyzer/files"


real_data=("directSpeech-to-directSpeech.csv" "directSpeech_wvoc.csv" "indirectSpeech-to-directSpeech.csv" "indirectSpeech-to-indirectSpeech.csv" "indirectSpeech_wvoc.csv" )
fake_data=("directSpeech-to-directSpeech.csv" "directSpeech_wvoc.csv" "indirectSpeech-to-directSpeech.csv" "indirectSpeech-to-indirectSpeech.csv" "indirectSpeech_wvoc.csv" )


config_real=("DS2DS" "VocDS" "IS2DS" "IS2IS" "VocIS")
config_fake=("DS2DS" "VocDS" "IS2DS" "IS2IS" "VocIS")







for i in "${!real_data[@]}"  ;do
	realFpath=`echo ${data_dir}/${real_data[i]}`
	realConfig=`echo ${config_real[i]}`
	for j in "${!fake_data[@]}"  ;do
		fakeFpath=`echo ${data_dir}/${fake_data[j]}`
		fakeConfig=`echo ${config_fake[j]}`
		if [ -f $realFpath ] && [ -f $fakeFpath ]  && [ ${realConfig} != ${fakeConfig}  ]  ;then

			printf " Real: %s \t   Fake  %s\n" "`echo ${realConfig}`" "`echo ${fakeConfig}`" >> result_intraSpk.log
			python converted_speech_similarity.py ${realFpath} ${fakeFpath} --realConfig ${realConfig} --fakeConfig ${fakeConfig} >> result_intraSpk.log
		fi
	done
done
