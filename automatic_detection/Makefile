TAG 			:= latest	
USER 			:= tanchavez
PROJECT			:= block-detection

IMG_WEB_SVC    		:= ${USER}/${PROJECT}:${TAG}
IMG_WEB_SVC_JYP    		:= ${USER}/${PROJECT_JYP}:${TAG}
.PHONY:

test:
	echo ${IMG_WEB_SVC}
	echo ${TAG}
	echo ${PROJECT}
	echo ${PROJECT}:${TAG}

build_docker: 
	docker build -t ${IMG_WEB_SVC} -f ./docker/Dockerfile .
	
run_docker:
	docker run --mount source=${PWD}/data,target=/data,type=bind ${USER}/${PROJECT} python3 block_detection.py /data/input_data /data/results

push_docker:
	docker push ${IMG_WEB_SVC}
clean: 
	find -name "*~" -delete
	-rm .python_history
	-rm -rf .config
	-rm -rf .cache
