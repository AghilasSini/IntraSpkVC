# input
# Stimuli Files 
#   - 
data_dir="/mnt/mydrive/home/aghilas/Workspace/Experiments/SpeakerCoding/Resemblyzer/files"


real_data=("interGender_conv_mode.csv" "intraGender_conv_mode.csv" "synt_mode.csv" "ref_neuronalVocoder_mode.csv" )

config_real=("ConvInterGen" "ConvIntraGen" "TTS" "VocTargetVoice")


fake_data=("interGender_conv_mode.csv" "interGenderSource_voc_mode.csv" "intraGender_conv_mode.csv" "intraGenderSource_voc_mode.csv" "synt_mode.csv"  )



config_fake=("VConv" "Clone" "Vocoded"  "TTS" )

for i in "${!real_data[@]}"  ;do
	realFpath=`echo ${data_dir}/${real_data[i]}`
	realConfig=`echo ${config_real[i]}`
	for j in "${!fake_data[@]}"  ;do
		fakeFpath=`echo ${data_dir}/${fake_data[j]}`
		fakeConfig=`echo ${config_fake[j]}`
		if [ -f $realFpath ] && [ -f $fakeFpath ]  && [ ${realConfig} != ${fakeConfig}  ]  ;then

			printf " Real: %s \t   Fake  %s\n" "`echo ${realConfig}`" "`echo ${fakeConfig}`" >> result.log
			python converted_speech_similarity.py ${realFpath} ${fakeFpath} --realConfig ${realConfig} --fakeConfig ${fakeConfig} >> result.log
		fi
	done
done
