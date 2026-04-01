source /opt/anaconda3/etc/profile.d/conda.sh
conda activate lattisense
export GOWORK=off
go run main.go llama.go linear.go nonlinear_moai.go nonlinear_thor.go util.go -test=Ops -logN=16 -parallel=true
