#python3 Train_Net.py --resume False  --network smallresnet --loss BCE
#python3 Train_Net.py --resume False  --network smallresnet --loss EXP_BCE
#python3 Train_Net.py --resume False  --network smallresnet --loss focal
#python3 Train_Net.py --resume False  --network smallresnet --loss EXPBCE_Focal_Balanced
#python3 Train_Net.py --resume False  --network smallresnet --loss EXPBCE_BCE_Balanced

python3 Train_Net.py --resume False  --network densenet --loss BCE
python3 Train_Net.py --resume False  --network densenet --loss EXP_BCE
python3 Train_Net.py --resume False  --network densenet --loss focal
python3 Train_Net.py --resume False  --network densenet --loss EXPBCE_Focal_Balanced
python3 Train_Net.py --resume False  --network densenet --loss EXPBCE_BCE_Balanced


#python3 Train_Net.py  --network densenet --loss BCE --lrDecay 0.99

#python3 Train_Net.py  --network densenet --loss BCE --lrDecay 0.98

#python3 Train_Net.py  --network densenet --loss BCE --lrDecay 0.97

#python3 Train_Net.py  --network densenet --loss BCE --lrDecay 0.96


#python3 Train_Net.py  --network densenet --loss EXP_BCE --lrDecay 0.99

#python3 Train_Net.py  --network densenet --loss EXP_BCE --lrDecay 0.98

#python3 Train_Net.py  --network densenet --loss EXP_BCE --lrDecay 0.97

#python3 Train_Net.py  --network densenet --loss EXP_BCE --lrDecay 0.96


#python3 Train_Net.py  --network densenet --loss focal --lrDecay 0.99

#python3 Train_Net.py  --network densenet --loss focal --lrDecay 0.98

#python3 Train_Net.py  --network densenet --loss focal --lrDecay 0.97

#python3 Train_Net.py  --network densenet --loss focal--lrDecay 0.96


#python3 Train_Net.py  --network densenet --loss EXPBCE_Focal_Balanced --lrDecay 0.99

#python3 Train_Net.py  --network densenet --loss EXPBCE_Focal_Balanced --lrDecay 0.98
#python3 Train_Net.py  --network densenet --loss EXPBCE_Focal_Balanced --lrDecay 0.97

#python3 Train_Net.py  --network densenet --loss EXPBCE_Focal_Balanced --lrDecay 0.96


#python3 Train_Net.py  --network densenet --loss EXPBCE_BCE_Balanced --lrDecay 0.99

#python3 Train_Net.py  --network densenet --loss EXPBCE_BCE_Balanced --lrDecay 0.98

#python3 Train_Net.py  --network densenet --loss EXPBCE_BCE_Balanced --lrDecay 0.97

#python3 Train_Net.py  --network densenet --loss EXPBCE_BCE_Balanced --lrDecay 0.96

