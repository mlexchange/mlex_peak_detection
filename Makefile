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
	docker run --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 --memory-swap -1 -it -v ${PWD}:/app/work/ -v ${PWD}/../data:/app/data -p 8052:8052 ${IMG_WEB_SVC}

push_docker:
	docker push ${IMG_WEB_SVC}
clean: 
	find -name "*~" -delete
	-rm .python_history
	-rm -rf .config
	-rm -rf .cache
